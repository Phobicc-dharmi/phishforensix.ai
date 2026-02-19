"""
╔══════════════════════════════════════════════════════════════╗
║         PhishForensix — Phishing Detection ML Model          ║
║         Random Forest Classifier · 30 URL Features           ║
║         Ready to integrate with Flask backend                 ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python phish_model.py               → Train & save model
    python phish_model.py --test        → Run test predictions
    python phish_model.py --predict <url> → Predict single URL
"""

import re
import math
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
#  SECTION 1 — FEATURE EXTRACTION ENGINE
#  Extracts 30 numerical features from any URL
# ══════════════════════════════════════════════════════════

# Common phishing keywords found in malicious URLs
PHISHING_KEYWORDS = [
    'secure', 'account', 'update', 'confirm', 'verify', 'login',
    'signin', 'banking', 'paypal', 'sbi', 'hdfc', 'icici', 'upi',
    'paytm', 'refund', 'claim', 'kyc', 'aadhaar', 'reward',
    'winner', 'alert', 'suspend', 'blocked', 'expire', 'urgent',
    'limited', 'offer', 'free', 'prize', 'click', 'here'
]

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = [
    '.xyz', '.top', '.online', '.site', '.click', '.link',
    '.info', '.biz', '.tk', '.ml', '.ga', '.cf', '.gq',
    '.pw', '.cc', '.icu', '.live', '.fun', '.shop'
]

# Legitimate popular domains (won't be flagged)
LEGITIMATE_DOMAINS = [
    'google', 'facebook', 'youtube', 'amazon', 'wikipedia',
    'twitter', 'instagram', 'linkedin', 'microsoft', 'apple',
    'github', 'stackoverflow', 'reddit', 'netflix', 'flipkart',
    'myntra', 'zomato', 'swiggy', 'irctc', 'sbi', 'hdfc'
]


def extract_features(url: str) -> dict:
    """
    Extract 30 numerical features from a URL for phishing detection.
    Returns a dictionary of feature_name: value pairs.
    """
    features = {}

    # Ensure URL has a scheme for parsing
    if not url.startswith(('http://', 'https://')):
        url_parsed = urlparse('http://' + url)
    else:
        url_parsed = urlparse(url)

    full_url    = url
    domain      = url_parsed.netloc.lower()
    path        = url_parsed.path.lower()
    scheme      = url_parsed.scheme.lower()
    query       = url_parsed.query.lower()
    hostname    = domain.split(':')[0]  # strip port

    # ── LEXICAL FEATURES ──────────────────────────────────

    # 1. Total URL length
    features['url_length'] = len(full_url)

    # 2. Domain length
    features['domain_length'] = len(hostname)

    # 3. Number of dots in URL
    features['num_dots'] = full_url.count('.')

    # 4. Number of hyphens in domain
    features['num_hyphens'] = hostname.count('-')

    # 5. Number of underscores in URL
    features['num_underscores'] = full_url.count('_')

    # 6. Number of slashes in URL (beyond protocol)
    features['num_slashes'] = path.count('/')

    # 7. Number of @ symbols (common in phishing to mislead)
    features['num_at'] = full_url.count('@')

    # 8. Number of question marks
    features['num_question_marks'] = full_url.count('?')

    # 9. Number of equals signs
    features['num_equals'] = full_url.count('=')

    # 10. Number of ampersands
    features['num_ampersands'] = full_url.count('&')

    # 11. Number of percent encodings
    features['num_percent'] = full_url.count('%')

    # 12. Number of digits in domain
    features['num_digits_in_domain'] = sum(c.isdigit() for c in hostname)

    # 13. Digit to letter ratio in URL
    digits  = sum(c.isdigit() for c in full_url)
    letters = sum(c.isalpha() for c in full_url)
    features['digit_letter_ratio'] = round(digits / max(letters, 1), 4)

    # 14. Phishing keyword count
    url_lower = full_url.lower()
    features['phishing_keyword_count'] = sum(
        1 for kw in PHISHING_KEYWORDS if kw in url_lower
    )

    # 15. Uses HTTPS (1 = yes, 0 = no)
    features['has_https'] = 1 if scheme == 'https' else 0

    # ── STRUCTURAL FEATURES ───────────────────────────────

    # 16. Number of subdomains
    parts = hostname.split('.')
    features['num_subdomains'] = max(0, len(parts) - 2)

    # 17. Has IP address as domain
    ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    features['has_ip_address'] = 1 if re.match(ip_pattern, hostname) else 0

    # 18. URL path depth
    features['path_depth'] = len([p for p in path.split('/') if p])

    # 19. Has port number
    features['has_port'] = 1 if ':' in domain and not domain.endswith(':80') and not domain.endswith(':443') else 0

    # 20. Suspicious TLD
    features['suspicious_tld'] = 1 if any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0

    # 21. Domain has number prefix/suffix
    features['domain_has_number'] = 1 if re.search(r'\d', hostname) else 0

    # 22. URL has redirect pattern (common in phishing)
    redirect_patterns = ['redirect', 'redir', 'forward', 'goto', 'url=', 'link=', 'next=']
    features['has_redirect'] = 1 if any(p in url_lower for p in redirect_patterns) else 0

    # 23. Has double slash in path (abnormal)
    features['has_double_slash'] = 1 if '//' in path else 0

    # 24. Query string length
    features['query_length'] = len(query)

    # 25. Number of query parameters
    features['num_query_params'] = len(query.split('&')) if query else 0

    # ── HEURISTIC FEATURES ────────────────────────────────

    # 26. URL entropy (randomness — high entropy = likely generated domain)
    def shannon_entropy(s):
        if not s:
            return 0
        prob = [float(s.count(c)) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in prob if p > 0)

    features['url_entropy'] = round(shannon_entropy(hostname), 4)

    # 27. Legitimate brand in subdomain (brand squatting)
    legitimate_brands = ['paypal', 'google', 'facebook', 'amazon', 'apple',
                         'microsoft', 'sbi', 'hdfc', 'icici', 'paytm', 'irctc']
    subdomain_part = '.'.join(parts[:-2]) if len(parts) > 2 else ''
    features['brand_in_subdomain'] = 1 if any(b in subdomain_part for b in legitimate_brands) else 0

    # 28. Longest word length in domain (very long words = suspicious)
    words = re.split(r'[.\-_]', hostname)
    features['longest_word_length'] = max((len(w) for w in words), default=0)

    # 29. Has suspicious file extension in path
    suspicious_exts = ['.exe', '.php', '.asp', '.jsp', '.cgi', '.pl']
    features['suspicious_extension'] = 1 if any(path.endswith(ext) for ext in suspicious_exts) else 0

    # 30. Is known legitimate domain (whitelist check)
    features['is_known_legitimate'] = 1 if any(
        leg in hostname for leg in LEGITIMATE_DOMAINS
    ) and features['num_subdomains'] == 0 else 0

    return features


def features_to_array(url: str) -> np.ndarray:
    """Convert URL features dict to numpy array for model input."""
    feat = extract_features(url)
    return np.array(list(feat.values())).reshape(1, -1)


def get_feature_names() -> list:
    """Return ordered list of feature names."""
    dummy = extract_features("http://example.com")
    return list(dummy.keys())


# ══════════════════════════════════════════════════════════
#  SECTION 2 — SYNTHETIC DATASET GENERATOR
#  Generates realistic training data since we can't
#  download PhishTank/OpenPhish at runtime.
#  In production: replace with real PhishTank CSV data.
# ══════════════════════════════════════════════════════════

def generate_training_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate a balanced synthetic dataset of phishing and legitimate URLs.
    Features are engineered to reflect real-world URL patterns.
    """
    np.random.seed(42)
    records = []

    # ── PHISHING URL PATTERNS ──
    phishing_templates = [
        "http://secure-{bank}-verify.{tld}/auth/login",
        "http://{bank}-kyc-update.{tld}/verify-now",
        "http://www.{bank}-blocked-account.{tld}/secure",
        "http://{bank}.account-verify.{tld}/user/signin",
        "http://{brand}-refund-claim.{tld}/process",
        "http://update-{bank}-account.{tld}/login",
        "http://{brand}.free-reward.{tld}/claim/{id}",
        "http://192.168.{a}.{b}/phish/login",
        "http://{bank}-official-update.{tld}/kyc/verify?id={id}&token={token}",
        "http://secure.{bank}-netbanking-{tld}.com/login",
    ]

    banks  = ['sbi','hdfc','icici','axis','paytm','irctc','aadhaar','upi','gov']
    brands = ['amazon','flipkart','paypal','google','apple','microsoft']
    tlds   = ['xyz','top','online','site','click','tk','ml','info','pw','icu']

    n_phish = n_samples // 2

    for _ in range(n_phish):
        bank  = np.random.choice(banks)
        brand = np.random.choice(brands)
        tld   = np.random.choice(tlds)
        url   = np.random.choice(phishing_templates).format(
            bank=bank, brand=brand, tld=tld,
            a=np.random.randint(1,255), b=np.random.randint(1,255),
            id=np.random.randint(1000,9999),
            token=''.join(np.random.choice(list('abcdef0123456789'), 16))
        )
        feat = extract_features(url)
        feat['label'] = 1  # phishing
        records.append(feat)

    # ── LEGITIMATE URL PATTERNS ──
    legit_templates = [
        "https://www.{domain}.com",
        "https://{domain}.com/{path}",
        "https://www.{domain}.in",
        "https://{domain}.org/{path}",
        "https://mail.{domain}.com",
        "https://docs.{domain}.com/{path}",
        "https://support.{domain}.com",
        "https://www.{domain}.co.in/{path}",
    ]

    legit_domains = [
        'google','facebook','amazon','flipkart','youtube',
        'github','twitter','linkedin','microsoft','apple',
        'stackoverflow','wikipedia','reddit','netflix','zomato',
        'swiggy','myntra','paytm','phonepe','gpay','indianrailways',
        'irctc','sbi','hdfcbank','icicibank','axisbank'
    ]

    paths = ['home','about','products','services','account','help',
             'contact','blog','news','support','login','register']

    n_legit = n_samples - n_phish

    for _ in range(n_legit):
        domain = np.random.choice(legit_domains)
        path   = np.random.choice(paths)
        url    = np.random.choice(legit_templates).format(
            domain=domain, path=path
        )
        feat = extract_features(url)
        feat['label'] = 0  # legitimate
        records.append(feat)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  ✓ Dataset generated: {len(df)} samples "
          f"({df['label'].sum()} phishing, {(df['label']==0).sum()} legitimate)")
    return df


# ══════════════════════════════════════════════════════════
#  SECTION 3 — MODEL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════

def train_model(n_samples: int = 5000, save_path: str = 'phish_model.pkl'):
    """
    Full training pipeline:
    1. Generate / load dataset
    2. Extract features
    3. Train Random Forest
    4. Evaluate & print metrics
    5. Save model + scaler to pickle
    """
    print("\n" + "═"*60)
    print("  PhishForensix · ML Training Pipeline")
    print("═"*60)

    # ── Step 1: Data ──
    print("\n[1/5] Generating training dataset...")
    df = generate_training_data(n_samples)

    feature_names = get_feature_names()
    X = df[feature_names].values
    y = df['label'].values

    # ── Step 2: Split ──
    print("\n[2/5] Splitting dataset (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Step 3: Scale ──
    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Step 4: Train ──
    print("\n[4/5] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sc, y_train)
    print("  ✓ Model trained successfully")

    # ── Step 5: Evaluate ──
    print("\n[5/5] Evaluating model...")
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print("\n  ┌─────────────────────────────┐")
    print(f"  │  Accuracy  : {acc*100:.2f}%           │")
    print(f"  │  Precision : {prec*100:.2f}%           │")
    print(f"  │  Recall    : {rec*100:.2f}%           │")
    print(f"  │  F1 Score  : {f1*100:.2f}%           │")
    print("  └─────────────────────────────┘")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  ┌─────────────────────────────┐")
    print(f"  │  TN={cm[0][0]:4d}  │  FP={cm[0][1]:4d}        │")
    print(f"  │  FN={cm[1][0]:4d}  │  TP={cm[1][1]:4d}        │")
    print(f"  └─────────────────────────────┘")

    # Cross validation
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Top 10 feature importances
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Top 10 Feature Importances:")
    for i, (fname, imp) in enumerate(feat_imp[:10], 1):
        bar = '█' * int(imp * 200)
        print(f"  {i:2}. {fname:<30} {imp:.4f}  {bar}")

    # ── Save ──
    bundle = {
        'model':         model,
        'scaler':        scaler,
        'feature_names': feature_names,
        'metrics': {
            'accuracy':  round(acc, 4),
            'precision': round(prec, 4),
            'recall':    round(rec, 4),
            'f1':        round(f1, 4),
        },
        'version': '1.0',
        'n_features': len(feature_names),
    }

    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)

    print(f"\n  ✓ Model saved → {save_path}")
    print("═"*60 + "\n")

    return bundle


# ══════════════════════════════════════════════════════════
#  SECTION 4 — PREDICTION ENGINE
#  This is what Flask will call for each URL scan
# ══════════════════════════════════════════════════════════

def load_model(model_path: str = 'phish_model.pkl') -> dict:
    """Load saved model bundle from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_url(url: str, model_bundle: dict = None, model_path: str = 'phish_model.pkl') -> dict:
    """
    Predict whether a URL is phishing or legitimate.

    Returns:
        {
            'url':         str,
            'verdict':     'PHISHING' | 'SUSPICIOUS' | 'SAFE',
            'risk_score':  int (0–100),
            'confidence':  float (0–1),
            'label':       1 | 0,
            'features':    dict of all 30 extracted features,
            'top_reasons': list of top suspicious feature explanations
        }
    """
    if model_bundle is None:
        model_bundle = load_model(model_path)

    model   = model_bundle['model']
    scaler  = model_bundle['scaler']

    # Extract features
    features = extract_features(url)
    X = np.array(list(features.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Predict
    label      = model.predict(X_scaled)[0]
    proba      = model.predict_proba(X_scaled)[0]
    confidence = float(proba[1])  # probability of being phishing
    risk_score = int(confidence * 100)

    # Determine verdict
    if confidence >= 0.75:
        verdict = 'PHISHING'
    elif confidence >= 0.40:
        verdict = 'SUSPICIOUS'
    else:
        verdict = 'SAFE'

    # Generate human-readable reasons
    top_reasons = _generate_reasons(features, verdict)

    return {
        'url':        url,
        'verdict':    verdict,
        'risk_score': risk_score,
        'confidence': round(confidence, 4),
        'label':      int(label),
        'features':   features,
        'top_reasons': top_reasons,
    }


def _generate_reasons(features: dict, verdict: str) -> list:
    """Generate human-readable explanations for the prediction."""
    reasons = []

    if features['phishing_keyword_count'] >= 2:
        reasons.append(f"Contains {features['phishing_keyword_count']} phishing keywords (e.g. verify, secure, update)")
    if features['num_subdomains'] >= 2:
        reasons.append(f"Suspicious subdomain structure ({features['num_subdomains']} subdomains detected)")
    if features['url_length'] > 60:
        reasons.append(f"Abnormally long URL ({features['url_length']} characters)")
    if features['has_ip_address']:
        reasons.append("URL uses raw IP address instead of domain name")
    if features['num_hyphens'] >= 2:
        reasons.append(f"Multiple hyphens in domain ({features['num_hyphens']} hyphens) — brand impersonation pattern")
    if not features['has_https']:
        reasons.append("No HTTPS — connection is not encrypted")
    if features['suspicious_tld']:
        reasons.append("Suspicious top-level domain (.xyz, .top, .online, etc.)")
    if features['brand_in_subdomain']:
        reasons.append("Legitimate brand name used in subdomain — possible brand squatting")
    if features['url_entropy'] > 3.8:
        reasons.append(f"High URL entropy ({features['url_entropy']:.2f}) — domain may be auto-generated")
    if features['num_at'] > 0:
        reasons.append("URL contains '@' symbol — used to deceive users about destination")
    if features['has_redirect']:
        reasons.append("URL contains redirect pattern — may forward to malicious page")
    if features['num_digits_in_domain'] >= 3:
        reasons.append(f"Domain contains {features['num_digits_in_domain']} digits — unusual for legitimate sites")

    if not reasons and verdict == 'SAFE':
        reasons.append("No major phishing indicators detected")
        if features['has_https']:
            reasons.append("HTTPS encryption present")
        if features['is_known_legitimate']:
            reasons.append("Domain matches known legitimate website")

    return reasons[:5]  # return top 5 reasons


# ══════════════════════════════════════════════════════════
#  SECTION 5 — FLASK-READY API FUNCTION
#  Copy this into your app.py
# ══════════════════════════════════════════════════════════

FLASK_INTEGRATION_CODE = '''
# ─────────────────────────────────────────────────────────
# ADD THIS TO YOUR app.py (Flask Backend)
# ─────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
from phish_model import predict_url, load_model
import os

app = Flask(__name__)

# Load model once at startup
MODEL = load_model('phish_model.pkl')

@app.route('/api/scan', methods=['POST'])
def scan_url():
    """
    POST /api/scan
    Body: { "url": "http://suspicious-site.com" }
    Returns: verdict, risk_score, features, reasons
    """
    data = request.get_json()
    url  = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    result = predict_url(url, MODEL)

    return jsonify({
        'url':        result['url'],
        'verdict':    result['verdict'],
        'risk_score': result['risk_score'],
        'confidence': result['confidence'],
        'features':   result['features'],
        'reasons':    result['top_reasons'],
    })

# ─────────────────────────────────────────────────────────
# FRONTEND JAVASCRIPT — paste in scanner.html
# ─────────────────────────────────────────────────────────
# Replace the fake detectProfile() function with:
#
# async function startScan() {
#   const url = document.getElementById('urlInput').value.trim();
#   if (!url) return;
#   // show progress...
#   const res  = await fetch('/api/scan', {
#     method: 'POST',
#     headers: { 'Content-Type': 'application/json' },
#     body: JSON.stringify({ url })
#   });
#   const data = await res.json();
#   showResults(data);  // pass real data to your UI
# }
# ─────────────────────────────────────────────────────────
'''


# ══════════════════════════════════════════════════════════
#  SECTION 6 — TEST SUITE
# ══════════════════════════════════════════════════════════

TEST_URLS = [
    # Known phishing
    ("http://secure-hdfc-verify.phish-login.com/auth/login",     "PHISHING"),
    ("http://paytm-kyc-update.online/verify-now",                 "PHISHING"),
    ("http://irctc-refund-claim.xyz/process/verify",              "PHISHING"),
    ("http://192.168.1.1/bank/login.php",                         "PHISHING"),
    ("http://sbi-account-blocked.top/secure/login?id=1234",       "PHISHING"),
    ("http://aadhaar-link-now.xyz/verify?token=abc123",           "PHISHING"),
    ("http://pm-kisan-apply.net/form/claim?reward=5000",          "PHISHING"),
    # Suspicious
    ("http://amazon-offer-india.store/deals",                     "SUSPICIOUS"),
    ("http://hdfc-loan-offer.in/apply-now",                       "SUSPICIOUS"),
    # Legitimate
    ("https://www.google.com",                                    "SAFE"),
    ("https://www.amazon.in/products",                            "SAFE"),
    ("https://github.com/user/repo",                              "SAFE"),
    ("https://www.sbi.co.in/web/personal-banking",                "SAFE"),
]


def run_tests(model_bundle: dict):
    """Run test predictions and print results."""
    print("\n" + "═"*60)
    print("  PhishForensix · Test Predictions")
    print("═"*60)

    correct = 0
    total   = len(TEST_URLS)

    for url, expected in TEST_URLS:
        result  = predict_url(url, model_bundle)
        verdict = result['verdict']
        score   = result['risk_score']

        # For test pass/fail: SUSPICIOUS counts as phishing-side
        predicted_class = 'PHISHING' if verdict in ('PHISHING', 'SUSPICIOUS') else 'SAFE'
        expected_class  = 'PHISHING' if expected in ('PHISHING', 'SUSPICIOUS') else 'SAFE'
        is_correct      = predicted_class == expected_class
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        color_verdict = verdict

        short_url = url[:55] + '...' if len(url) > 55 else url
        print(f"\n  {status} [{score:3d}%] {color_verdict:<10} → {short_url}")
        if result['top_reasons']:
            print(f"     ↳ {result['top_reasons'][0]}")

    acc = correct / total * 100
    print(f"\n  ─────────────────────────────────────────")
    print(f"  Test Accuracy: {correct}/{total} ({acc:.0f}%)")
    print("═"*60 + "\n")


# ══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PhishForensix ML Model')
    parser.add_argument('--test',    action='store_true', help='Run test suite')
    parser.add_argument('--predict', type=str,            help='Predict a single URL')
    parser.add_argument('--samples', type=int, default=5000, help='Training dataset size')
    parser.add_argument('--model',   type=str, default='phish_model.pkl', help='Model file path')
    args = parser.parse_args()

    if args.predict:
        # Just predict without training
        try:
            bundle = load_model(args.model)
            result = predict_url(args.predict, bundle)
            print(f"\n  URL     : {result['url']}")
            print(f"  Verdict : {result['verdict']}")
            print(f"  Risk    : {result['risk_score']}%")
            print(f"  Reasons :")
            for r in result['top_reasons']:
                print(f"    · {r}")
        except FileNotFoundError:
            print(f"\n  ⚠ Model not found. Run without --predict first to train.")

    elif args.test:
        try:
            bundle = load_model(args.model)
            print(f"\n  ✓ Loaded existing model from {args.model}")
        except FileNotFoundError:
            print(f"\n  ℹ Model not found — training first...")
            bundle = train_model(args.samples, args.model)
        run_tests(bundle)

    else:
        # Default: train and save
        bundle = train_model(args.samples, args.model)
        print("\n  Running quick test on trained model...")
        run_tests(bundle)
        print(FLASK_INTEGRATION_CODE)
