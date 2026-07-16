"""
ETL for the 2023 Equibase U.S. Thoroughbred dataset.

Two raw sources:
  - "2023 Result Charts.zip"  -> TRKyyyymmddtch.xml   (post-race charts)
  - "2023 PPs.zip"            -> SIMDyyyymmddTRK_CTR.zip -> .xml (pre-race entry/PP files; used only
                                  for morning-line odds, which Result Charts do not carry pre-race)

Filtering rules (per task framing):
  - Thoroughbred only (BREED == 'TB' at the RACE level)
  - USA tracks located in one of the 50 states (Country == 'USA' in the Equibase Track codes sheet;
    this already excludes Puerto Rico [country 'PR'] and Canada [country 'CAN'])

Leakage discipline:
  - Every column pulled from a chart ENTRY is tagged either as a *pre-race* fact (known before that
    race went off: post position, weight, equipment, medication, jockey/trainer assignment, claimed
    price, race conditions) or a *post-race outcome* fact (finish position, running line, speed
    rating, final odds, payoffs). Post-race fields are prefixed `post_` and must only ever be used to
    build historical features for a LATER race for the same horse/jockey/trainer -- never as a feature
    of the row's own race.
"""
import re
import zipfile
import io
from datetime import date

from lxml import etree
import pandas as pd
import openpyxl

CHART_NAME_RE = re.compile(r'^([A-Za-z0-9]+)(\d{8})tch\.xml$')
PP_NAME_RE = re.compile(r'^SIMD(\d{8})([A-Za-z0-9]+)_([A-Za-z]{2,3})\.(?:xml\.)?zip$')


def load_usa_track_map(params_xlsx_path):
    wb = openpyxl.load_workbook(params_xlsx_path, read_only=True, data_only=True)
    ws = wb['Track codes']
    rows = list(ws.iter_rows(values_only=True))[1:]
    usa = {}
    for country, code, desc, state in rows:
        if country == 'USA':
            usa[code.upper()] = {'state': state, 'name': desc}
    return usa


def _text(el, tag, default=None):
    child = el.find(tag)
    if child is None or child.text is None:
        return default
    t = child.text.strip()
    return t if t else default


def _num(el, tag, default=None):
    v = _text(el, tag)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _distance_to_furlongs(distance_val, unit):
    if distance_val is None:
        return None
    if unit == 'F':
        return distance_val / 100.0
    if unit == 'Y':
        return distance_val / 220.0
    return None


RACE_CLASS_KEYWORDS = [
    ('Stakes', ['STAKES', 'HANDICAP']),
    ('Maiden Claiming', ['MAIDEN CLAIMING', 'MCL']),
    ('Maiden Special Weight', ['MAIDEN SPECIAL', 'MAIDEN']),
    ('Starter Allowance', ['STARTER']),
    ('Optional Claiming', ['OPTIONAL CLAIMING', 'OC']),
    ('Allowance', ['ALLOWANCE']),
    ('Claiming', ['CLAIMING', 'CLM']),
]


def classify_race_class(type_text):
    if not type_text:
        return 'Other'
    up = type_text.upper()
    for label, keywords in RACE_CLASS_KEYWORDS:
        if any(k in up for k in keywords):
            return label
    return 'Other'


def parse_chart_xml(xml_bytes, track_code, file_date_str):
    root = etree.fromstring(xml_bytes)
    rows = []
    for race in root.findall('RACE'):
        breed = _text(race, 'BREED')
        if breed != 'TB':
            continue
        race_number = int(race.get('NUMBER'))
        distance_val = _num(race, 'DISTANCE')
        dist_unit = _text(race, 'DIST_UNIT')
        distance_furlongs = _distance_to_furlongs(distance_val, dist_unit)
        type_text = _text(race, 'TYPE')
        entries = race.findall('ENTRY')
        field_size = len(entries)
        base_race = {
            'date': file_date_str,
            'track_code': track_code,
            'race_number': race_number,
            'breed': breed,
            'race_type_text': type_text,
            'race_class': classify_race_class(type_text),
            'age_restrictions': _text(race, 'AGE_RESTRICTIONS'),
            'purse': _num(race, 'PURSE'),
            'distance_furlongs': distance_furlongs,
            'about_dist_flag': _text(race, 'ABOUT_DIST_FLAG'),
            'course_id': _text(race, 'COURSE_ID'),
            'surface': _text(race, 'SURFACE'),
            'class_rating': _num(race, 'CLASS_RATING'),
            'track_condition': _text(race, 'TRK_COND'),
            'weather': _text(race, 'WEATHER'),
            'post_time': _text(race, 'POST_TIME'),
            'field_size': field_size,
        }
        for entry in entries:
            jockey = entry.find('JOCKEY')
            trainer = entry.find('TRAINER')
            poc_final = None
            for poc in entry.findall('POINT_OF_CALL'):
                if poc.get('WHICH') == 'FINAL':
                    poc_final = poc
                    break
            row = dict(base_race)
            row.update({
                'horse_name': _text(entry, 'NAME'),
                'program_num': _text(entry, 'PROGRAM_NUM'),
                'post_pos': _num(entry, 'POST_POS'),
                'weight': _num(entry, 'WEIGHT'),
                'age': _num(entry, 'AGE'),
                'sex': _text(entry.find('SEX'), 'CODE') if entry.find('SEX') is not None else None,
                'meds': _text(entry, 'MEDS'),
                'equip': _text(entry, 'EQUIP'),
                'claim_price': _num(entry, 'CLAIM_PRICE'),
                'jockey_key': _text(jockey, 'KEY') if jockey is not None else None,
                'jockey_name': (
                    f"{_text(jockey, 'FIRST_NAME', '')} {_text(jockey, 'LAST_NAME', '')}".strip()
                    if jockey is not None else None
                ),
                'trainer_key': _text(trainer, 'KEY') if trainer is not None else None,
                'trainer_name': (
                    f"{_text(trainer, 'FIRST_NAME', '')} {_text(trainer, 'LAST_NAME', '')}".strip()
                    if trainer is not None else None
                ),
                # --- post-race outcome fields: usable only as HISTORY for later races ---
                'post_official_fin': _num(entry, 'OFFICIAL_FIN'),
                'post_speed_rating': _num(entry, 'SPEED_RATING'),
                'post_finish_time': _num(entry, 'FINISH_TIME'),
                'post_final_call_pos': _num(poc_final, 'POSITION') if poc_final is not None else None,
                'post_final_call_lengths': _num(poc_final, 'LENGTHS') if poc_final is not None else None,
                'post_dollar_odds': _num(entry, 'DOLLAR_ODDS'),
                'post_win_payoff': _num(entry, 'WIN_PAYOFF'),
            })
            rows.append(row)
    return rows


def parse_pp_xml(xml_bytes, track_code, file_date_str):
    root = etree.fromstring(xml_bytes)
    rows = []
    for race in root.findall('Race'):
        race_number_el = race.find('RaceNumber')
        if race_number_el is None or race_number_el.text is None:
            continue
        race_number = int(race_number_el.text)
        for starters in race.findall('Starters'):
            horse = starters.find('Horse')
            horse_name = _text(horse, 'HorseName') if horse is not None else None
            odds_text = _text(starters, 'Odds')
            rows.append({
                'date': file_date_str,
                'track_code': track_code,
                'race_number': race_number,
                'horse_name': horse_name,
                'program_num': _text(starters, 'ProgramNumber'),
                'morning_line_odds_text': odds_text,
            })
    return rows


def _frac_odds_to_decimal(text):
    """'20/1' -> 21.0 decimal odds (includes stake); '9/5' -> 2.8; returns None if unparseable."""
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if '/' in text:
        num, _, den = text.partition('/')
        try:
            num = float(num)
            den = float(den)
            if den == 0:
                return None
            return num / den + 1.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def build_chart_table(charts_zip_path, usa_track_map, log_every=1000):
    usa_codes = set(usa_track_map.keys())
    zf = zipfile.ZipFile(charts_zip_path)
    all_rows = []
    n_files = 0
    n_skipped_country = 0
    n_failed = 0
    names = [n for n in zf.namelist() if n.lower().endswith('tch.xml')]
    for i, name in enumerate(names):
        base = name.split('/')[-1]
        m = CHART_NAME_RE.match(base)
        if not m:
            continue
        track_code = m.group(1).upper()
        date_str = m.group(2)
        if track_code not in usa_codes:
            n_skipped_country += 1
            continue
        try:
            xml_bytes = zf.read(name)
            rows = parse_chart_xml(xml_bytes, track_code, date_str)
            all_rows.extend(rows)
            n_files += 1
        except Exception as e:
            n_failed += 1
        if log_every and (i + 1) % log_every == 0:
            print(f'  ...{i + 1}/{len(names)} chart files scanned, {len(all_rows)} entry-rows so far')
    print(f'Chart files parsed: {n_files}, skipped (non-USA track): {n_skipped_country}, failed: {n_failed}')
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df


def build_pp_table(pp_zip_path, usa_track_map, log_every=500):
    usa_codes = set(usa_track_map.keys())
    zf = zipfile.ZipFile(pp_zip_path)
    all_rows = []
    n_files = 0
    n_skipped_country = 0
    n_failed = 0
    names = [n for n in zf.namelist() if n.lower().endswith('.zip')]
    for i, name in enumerate(names):
        base = name.split('/')[-1]
        m = PP_NAME_RE.match(base)
        if not m:
            continue
        date_str, track_code, ctr = m.group(1), m.group(2).upper(), m.group(3).upper()
        if ctr != 'USA' or track_code not in usa_codes:
            n_skipped_country += 1
            continue
        try:
            inner_bytes = zf.read(name)
            inner_zf = zipfile.ZipFile(io.BytesIO(inner_bytes))
            xml_names = [n for n in inner_zf.namelist() if n.lower().endswith('.xml')]
            if not xml_names:
                continue
            xml_bytes = inner_zf.read(xml_names[0])
            rows = parse_pp_xml(xml_bytes, track_code, date_str)
            all_rows.extend(rows)
            n_files += 1
        except Exception:
            n_failed += 1
        if log_every and (i + 1) % log_every == 0:
            print(f'  ...{i + 1}/{len(names)} PP files scanned, {len(all_rows)} starter-rows so far')
    print(f'PP files parsed: {n_files}, skipped (non-USA track): {n_skipped_country}, failed: {n_failed}')
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.drop_duplicates(subset=['date', 'track_code', 'race_number', 'horse_name'], keep='first')
    df['morning_line_decimal_odds'] = df['morning_line_odds_text'].map(_frac_odds_to_decimal)
    df['morning_line_implied_prob'] = 1.0 / df['morning_line_decimal_odds']
    return df


if __name__ == '__main__':
    import time
    import os

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    PARAMS_XLSX = r'C:\Users\zmaje\Downloads\Equibase Dataset\Equibase Parameters.xlsx'
    CHARTS_ZIP = r'C:\Users\zmaje\Downloads\Equibase Dataset\2023 Result Charts.zip'
    PP_ZIP = r'C:\Users\zmaje\Downloads\Equibase Dataset\2023 PPs.zip'

    usa_map = load_usa_track_map(PARAMS_XLSX)
    print(f'Loaded {len(usa_map)} USA (50-state) track codes.')

    t0 = time.time()
    print('Parsing Result Charts (label + historical-feature source)...')
    chart_df = build_chart_table(CHARTS_ZIP, usa_map)
    print(f'chart_df shape={chart_df.shape}, elapsed={time.time() - t0:.1f}s')
    chart_df.to_parquet(os.path.join(DATA_DIR, 'chart_entries.parquet'), index=False)

    t0 = time.time()
    print('Parsing PP files (morning-line odds source)...')
    pp_df = build_pp_table(PP_ZIP, usa_map)
    print(f'pp_df shape={pp_df.shape}, elapsed={time.time() - t0:.1f}s')
    pp_df.to_parquet(os.path.join(DATA_DIR, 'pp_morning_lines.parquet'), index=False)

    print('Done. Saved to', DATA_DIR)
