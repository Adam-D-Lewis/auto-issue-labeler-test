#!/usr/bin/env python3
"""
Batch labeler: reads JSONL issues, queries OpenAI for label recommendations restricted
to a predefined set, and writes results to JSONL.

Usage:
  pixi run py scripts/batch_label_issues.py \
      --input scripts/data/export/issues.jsonl \
      --output scripts/data/export/generated-labels.jsonl \
      --model gpt-5 \
      --max-workers 4

Environment:
  - OPENROUTER_API_KEY (required)
  - OPENROUTER_TIMEOUT_SECS (optional, default 20)
  - OPENROUTER_MODEL (optional; overridden by --model if provided)

Notes:
  - Allowed label set mirrors scripts/label_issue.py
  - Robust to both Responses API and Chat Completions API, same approach as scripts/label_issue.py
"""

import argparse
import concurrent.futures
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
 
import requests
import yaml
 

keep_ids = ['1264585995', '1263826622', '1147285981', '1143035265', '1143992136', '1132659902', '1130650893', '1131451239', '1129733307', '1127520006', '1127524953', '1127510587', '1125008214', '1127512022', '1127253628', '1124483202', '1125008141', '1124480383', '1123390662', '1124582497', '1123557147', '1123352534', '1123340683', '1123318135', '1123299980', '1122320664', '1123311635', '1120936425', '1123338648', '1114653688', '1107206238', '1110683928', '1103450925', '1109842109', '1103899359', '1107200438', '1106326529', '1103239460', '1098025023', '1087186560', '1087139938', '1085837536', '1083667775', '1087143352', '1083531594', '1079962039', '1082629271', '1083465968', '1079993803', '1079908424', '1073225888', '1073669007', '1062649851', '1068612086', '1072392767', '1073216621', '1068469998', '1058990956', '1058981548', '1060533242', '1056273842', '1058980855', '1058683105', '1056258060', '1058400757', '1055473977', '1053394899', '1056842526', '1051036948', '1052168174', '1052177135', '1049955407', '1046461087', '1050259570', '1049501274', '1044748943', '1049124529', '1048754702', '1043396434', '1043061620', '1042748181', '1042635362', '1042586383', '1033995793', '1039399049', '1036411433', '1043393457', '1031620260', '1031443189', '1031551648', '1027593953', '1028125744', '1027601765', '1027651508', '1028083719', '1027566659', '1028209501', '1042590998', '1026314062', '1027462045', '1025848525', '1025296036', '1027124495', '1020014904', '1018707521', '1020992929', '1021411578', '1018307507', '1025776585', '1001523254', '1022544698', '1001565738', '996060313', '1002634889', '1012240051', '1000751243', '989843006', '1001066798', '995980722', '991475005', '985507451', '993272300', '986706238', '996068489', '983886078', '980446809', '980049527', '978095079', '978092618', '983885739', '978089252', '978086373', '979368762', '978084654', '978088332', '974918260', '978093401', '978090166', '974921263', '974915985', '974916645', '974917228', '974913039', '974910074', '974912157', '974901322', '974905135', '977968400', '974902539', '974907454', '972146542', '967234640', '972148754', '974898265', '974101976', '965149016', '967223106', '972992774', '966814346', '955130482', '964408211', '966755038', '955130416', '941004355', '955128119', '958895787', '944622201', '955024209', '945629147', '964135508', '955129046', '936134113', '940483015', '939213799', '934759243', '931686127', '934692922', '933579311', '943727681', '930165003', '931453246', '932919115', '938071669', '930207906', '934886023', '931694719', '928307961', '928675265', '931678676', '929277942', '927258846', '928568606', '927259987', '925204626', '927273420', '927743115', '923204344', '923220503', '924863692', '928202292', '922447080', '922435571', '922907191', '918947343', '922425423', '921631920', '921865727', '917986507', '922129557', '921446750', '922439735', '917436271', '917236610', '917576009', '917123732', '916362136', '917035583', '917393888', '913551690', '916225908', '907585206', '915190323', '904985837', '907582018', '908457959', '904985737', '900859975', '900863598', '1057244624', '910669018', '922662227', '903803541', '911503053', '900841789', '903519768', '900863916', '1024152612', '900859611', '900840725', '896743164', '900859467', '894852593', '895950461', '895615274', '898027912', '886551169', '891437960', '898153510', '899816972', '894434162', '888927137', '891053166', '885105973', '878936016', '879140951', '879253815', '879213684', '879064017', '891092881', '890228606', '894412118', '876731575', '877220729', '877219761', '876304451', '878018955', '876523738', '978083741', '876731542', '877224650', '875482614', '875742788', '877331779', '876236089', '875725287', '875591345', '871010481', '869795255', '870580812', '871013789', '876192856', '870018574', '866422739', '875746022', '863904300', '863780060', '869015673', '865547557', '869662014', '872303923', '862927915', '869786482', '862851697', '871054312', '865815652', '862794620', '865854924', '859096699', '861315397', '857914497', '862843603', '857353885', '859636140', '854590287', '859671332', '855983544', '860113287', '859954938', '855881593', '856090484', '856871305', '856009864', '857371412', '853909787', '849304738', '858373623', '854525938', '844449089', '849841438', '854283943', '843590929', '848594305', '849301700', '843426478', '844723689', '841208270', '844452470', '849841447', '850429461', '847012404', '843391138', '848500164', '840991637', '844570107', '841242495', '830332178', '842864436', '843285146', '838817054', '841136128', '834865907', '834835776', '834860427', '832849045', '840983143', '834826363', '830312000', '826096998', '832850115', '825010877', '829225980', '829215729', '828128690', '823233968', '838724252', '823021037', '931689942', '824798483', '826033897', '824670052', '822481926', '822213636', '826067353', '823455411', '824796471', '822210648', '822233658', '821506344', '821224548', '829224144', '822206093', '822605646', '821241119', '820425938', '819089548', '821241059', '817331693', '820116657', '819050341', '819039410', '820301734', '816504920', '816859960', '816506070', '816508952', '816542107', '819054026', '816512902', '820348897', '821224635', '816505573', '816505730', '813601593', '815349218', '808617397', '930160672', '816410855', '805749215', '803009979', '814571051', '811192147', '815349178', '811551201', '810503807', '810448002', '801413086', '799368001', '801278643', '796902967', '799361179', '808616159', '797055804', '805841166', '796244685', '804624865', '793414498', '793550818', '795199273', '793514278', '798450060', '793548770', '799346091', '795902318', '786990563', '798457816', '794475724', '793431195', '782215424', '791600565', '785501508', '786015013', '794341772', '795329056', '782166441', '787446422', '782847524', '792132247', '785447149', '782165810', '770203387', '778144987', '778320187', '782188949', '781329940', '770532532', '769027588', '770954953', '779616648', '785446963', '746320457', '769030686', '778311789', '742389312', '761887619', '761871311', '739977885', '764124823', '764128658', '754497480', '766613365', '742488915', '748908549', '742455641', '742486475', '744332691', '737770020', '739952948', '739139606', '739951289', '734662566', '739188000', '737002104', '739058521', '728950866', '737855814', '741974206', '729743644', '734588846', '735420204', '739950978', '736437070', '728652945', '736828870', '739141708', '736367838', '723413686', '728536279', '728407170', '733838922', '718221453', '726699712', '717433707', '728806689', '722458607', '728809409', '737685234', '723549243', '717417875', '711214358', '725676559', '714839648', '724901766', '728826897', '717455612', '709186659', '711178795', '717483569', '716600868', '711201431', '709249762', '710438046', '708512298', '711199153', '708341012', '708974588', '711193542', '711206926', '717387739', '708295656', '705714800', '716773012', '708382403', '706452107', '705732093', '705714609', '680677906', '705655086', '677785855', '680656596', '684687408', '872179122', '687491687', '680657045', '679814402', '705728971', '681110500', '705654175', '711187150', '680665641', '706458707', '680656781', '680656946', '680678380', '680665865', '680665538', '704473257', '679814491', '680668000', '680665408', '680667212', '680668811', '680667605', '680667925', '680668198', '680668137', '680667077', '680668703', '680665996', '680667686', '680667400', '680668259', '680668315', '680668908', '680667755', '680668972', '680669264', '680669362', '680669120', '596342409', '680669837', '680669992', '680669061', '680670133', '680668751', '680670071', '680669913', '680669650', '596340213', '827886434', '713711304', '781409054', '680668071', '680665725', '704519777', '926426308', '925011095', '816542855', '816516479']
 
def load_labels_from_yaml(path: str) -> Dict[str, str]:
    """
    Loads labels and descriptions from a YAML file formatted like .github/labels.yml
    Returns a dictionary of label_name: description
    """
    with open(path, "r", encoding="utf-8") as f:
        labels_data = yaml.safe_load(f)
 
    labels_map = {}
    if isinstance(labels_data, list):
        for item in labels_data:
            if isinstance(item, dict) and "name" in item and "description" in item:
                labels_map[item["name"]] = item["description"]
    return labels_map
 
 
ALLOWED_LABELS_WITH_DESC: Dict[str, str] = {}
 
VERSION = "batch-v1"
 
 
def log(msg: str) -> None:
    print(msg, flush=True)


def safe_int(v: Optional[str], default: int) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def build_openai_prompt(title: str, body: str, allowed_with_desc: Dict[str, str]) -> str:
    allowed_list_str = "\n".join(
        [f'- "{name}": {desc}' for name, desc in allowed_with_desc.items()]
    )
    return (
        "You are an assistant that assigns GitHub issue labels.\n"
        "Return ONLY a single line of comma-separated labels from the allowed set.\n"
        "Format example:\n"
        "bug, enhancement, documentation\n"
        "Rules:\n"
        "- Choose any number of labels from the allowed set (including zero).\n"
        "- Do not include any extra text, code fences, or explanations. Only the CSV line.\n\n"
        "Allowed labels with descriptions:\n"
        f"{allowed_list_str}\n\n"
        f"Issue title: {title}\n"
        f"Issue body:\n{body}\n"
    )


def call_openai(api_key: str, model: str, prompt: str, timeout_secs: int) -> str:
    """
    Calls OpenRouter to get plain text output (CSV string of labels only).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Use OpenRouter's Chat Completions endpoint
    url_chat = "https://openrouter.ai/api/v1/chat/completions"
    sys_prompt = (
        "You are an assistant that assigns GitHub issue labels. "
        "Return ONLY a single line, CSV of labels from the allowed set, no explanations."
    )
    data_chat = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    try:
        resp = requests.post(url_chat, headers=headers, json=data_chat, timeout=timeout_secs)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.post(url_chat, headers=headers, json=data_chat, timeout=timeout_secs)
        if resp.status_code // 100 != 2:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text}")
        out = resp.json()
        text = out["choices"][0]["message"]["content"]
        return text
    except requests.RequestException as e:
        raise RuntimeError(f"OpenRouter request error: {e}")


def parse_llm_labels_csv(s: str) -> List[str]:
    """
    Parse CSV returned by the LLM in the form:
    "bug, enhancement, documentation"
    - Ignores extra spaces and newlines
    - Accepts empty string (meaning no labels)
    - Returns a list of label names in order
    Mirrors scripts/label_issue.py
    """
    if not isinstance(s, str):
        s = str(s)
    text = s.strip()
    if not text:
        return []

    # Strip code fences or quotes
    if text.startswith("```") and text.endswith("```"):
        inner = text.strip("`")
        first_newline = inner.find("\n")
        if first_newline != -1:
            text = inner[first_newline + 1 :].strip()
        else:
            text = inner.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    items = [p.strip() for p in text.replace("\n", " ").split(",")]
    labels = [i for i in items if i]
    return labels


def select_labels_from_list(candidates: List[str], allowed: List[str]) -> List[str]:
    # Keep original order; filter against allowed
    return [name for name in candidates if name in allowed]


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield line_idx, obj
            except json.JSONDecodeError as e:
                log(f"Skipping invalid JSONL line {line_idx}: {e}")


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_issue(
    issue: Dict[str, Any],
    api_key: str,
    model: str,
    timeout_secs: int,
) -> Tuple[Any, List[str], Optional[str]]:
    """
    Returns: (id, labels, error_message_if_any)
    """
    issue_id = issue.get("id")
    title = issue.get("title") or ""
    body = issue.get("body") or ""

    try:
        prompt = build_openai_prompt(title, body, ALLOWED_LABELS_WITH_DESC)
        raw_text = call_openai(api_key, model, prompt, timeout_secs)
        label_list = parse_llm_labels_csv(raw_text)
        allowed_names = list(ALLOWED_LABELS_WITH_DESC.keys())
        chosen = select_labels_from_list(label_list, allowed_names)
        return (issue_id, chosen, None)
    except Exception as e:
        return (issue_id, [], str(e))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Batch label GitHub issues via OpenRouter and write JSONL outputs.")
    p.add_argument("--input", required=True, help="Path to input issues.jsonl")
    p.add_argument("--output", required=True, help="Path to write generated-labels.jsonl")
    p.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-pro"), help="OpenRouter model (default: google/gemini-2.5-pro)")
    p.add_argument("--max-workers", type=int, default=4, help="Concurrency for API calls (default: 4)")
    p.add_argument("--skip-errors", action="store_true", help="Do not fail on per-issue errors; record empty labels and continue")
    p.add_argument("--clobber", action="store_true", help="Overwrite output file if it exists and is not empty")
    p.add_argument(
        "--labels-yaml",
        default=".github/labels.yml",
        help="Path to YAML file with label names and descriptions (default: .github/labels.yml)",
    )
    args = p.parse_args(argv)
 
    global ALLOWED_LABELS_WITH_DESC
    try:
        ALLOWED_LABELS_WITH_DESC = load_labels_from_yaml(args.labels_yaml)
        if not ALLOWED_LABELS_WITH_DESC:
            log(f"Warning: No labels loaded from {args.labels_yaml}. Check file format.")
    except Exception as e:
        log(f"Error loading labels from {args.labels_yaml}: {e}")
        return 3
 
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        log("Error: OPENROUTER_API_KEY is required in environment.")
        return 2

    timeout_secs = safe_int(os.environ.get("OPENROUTER_TIMEOUT_SECS"), 20)

    input_path = args.input
    output_path = args.output

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Check output file before proceeding
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and not args.clobber:
        log(f"Error: Output file {output_path} already exists and is not empty. Use --clobber to overwrite.")
        return 4

    issues: List[Tuple[int, Dict[str, Any]]] = list(read_jsonl(input_path))
    log(f"Loaded {len(issues)} issues from {input_path}")
 
    errors: List[str] = []
 
    # Clear output file before starting
    with open(output_path, "w", encoding="utf-8") as f:
        pass

    # Process with limited concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = []
        for _, issue in issues:
            if issue.get("id") and str(issue.get("id")) in keep_ids:
                futures.append(
                    ex.submit(process_issue, issue, api_key, args.model, timeout_secs)
                )
 
        completed_count = 0
        total_count = len(issues)
        for fut in concurrent.futures.as_completed(futures):
            completed_count += 1
            issue_id, labels, err = fut.result()
            if err:
                msg = f"Issue {issue_id}: {err}"
                errors.append(msg)
                log(f"[WARN] {msg}")
            else:
                rec = {"id": issue_id, "labels": labels}
                append_jsonl(output_path, rec)
            
            log(f"Processed {completed_count}/{total_count} issues...")
 
    log(f"Wrote {completed_count} records to {output_path}")

    if errors and not args.skip_errors:
        log(f"Encountered {len(errors)} errors. Failing. Use --skip-errors to ignore.")
        return 1

    if errors:
        log(f"Completed with {len(errors)} errors (skipped).")
    else:
        log("Completed successfully with no errors.")

    return 0


if __name__ == "__main__":
    print(build_openai_prompt('title', 'body', load_labels_from_yaml('/home/balast/CodingProjects/auto-issue-labeler-test/.github/labels.yml')))
    sys.exit(main())