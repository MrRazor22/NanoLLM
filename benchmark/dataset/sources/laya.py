import json
from pathlib import Path
from typing import Any, Dict, List
from datasets import load_dataset
from benchmark.dataset.dataset import ADAPTED_DIR, IDataSource

class LayaSource(IDataSource):
    name = "laya"
    display_name = "4. Laya 6-Suite"
    category_labels: Dict[str, str] = {
        "jev.ag_news": "AG News",
        "jev.emotion": "DAIR Emotion",
        "jev.banking77_full": "Banking77 (77 classes)",
        "app.support_triage": "Support Triage",
        "app.email_spam": "Email Spam",
        "app.phishing": "Phishing",
    }

    def extract(self) -> List[Dict[str, Any]]:
        local_file = ADAPTED_DIR / "laya.json"
        if local_file.exists():
            with open(local_file, "r", encoding="utf-8") as f:
                return json.load(f)

        items: List[Dict[str, Any]] = []

        # 1. AG News
        ds_ag = load_dataset("fancyzhx/ag_news", split="test")
        ag_classes = ["world", "sports", "business", "sci_tech"]
        ag_criteria = {
            "world": "world news and international politics",
            "sports": "sports",
            "business": "business and economy",
            "sci_tech": "science and technology"
        }
        for row in ds_ag:
            if len([x for x in items if x["category"] == "jev.ag_news"]) >= 400: break
            lbl = int(row["label"])
            if 0 <= lbl < len(ag_classes):
                items.append({
                    "category": "jev.ag_news",
                    "state": f"Article: {str(row['text']).strip()}",
                    "questions": {"topic": {"type": "choice", "instructions": "What is the topic of article?", "criteria": ag_criteria}},
                    "gold": {"topic": {"type": "choice", "label": ag_classes[lbl]}}
                })

        # 2. Emotion
        ds_emo = load_dataset("dair-ai/emotion", "split", split="test")
        emo_classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        emo_criteria = {k: f"expressing {k}" for k in emo_classes}
        for row in ds_emo:
            if len([x for x in items if x["category"] == "jev.emotion"]) >= 400: break
            lbl = int(row["label"])
            if 0 <= lbl < len(emo_classes):
                items.append({
                    "category": "jev.emotion",
                    "state": f"Text: {str(row['text']).strip()}",
                    "questions": {"emotion": {"type": "choice", "instructions": "Which emotion is most strongly expressed in text?", "criteria": emo_criteria}},
                    "gold": {"emotion": {"type": "choice", "label": emo_classes[lbl]}}
                })

        # 3. Banking77
        ds_b77 = load_dataset("mteb/banking77", split="test")
        b77_opts = {str(n).replace("_", " "): str(n).replace("_", " ") for n in ds_b77.features["label"].names}
        for row in ds_b77:
            if len([x for x in items if x["category"] == "jev.banking77_full"]) >= 400: break
            lbl_name = str(row.get("label_text", "")).replace("_", " ")
            items.append({
                "category": "jev.banking77_full",
                "state": f"Message: {str(row['text']).strip()}",
                "questions": {"intent": {"type": "choice", "instructions": "What is the primary customer inquiry or banking request?", "criteria": b77_opts}},
                "gold": {"intent": {"type": "choice", "label": lbl_name}}
            })

        # 4. Support Triage
        ds_sup = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        sup_criteria = {
            "Technical Support": "technical problems, bugs, outages, integrations",
            "Product Support": "help using a product or feature",
            "Customer Service": "general account or service questions",
            "IT Support": "internal IT, devices, access, networks",
            "Billing and Payments": "invoices, charges, refunds, payment methods",
            "Returns and Exchanges": "returning or exchanging an item",
            "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
            "Sales and Pre-Sales": "pricing, quotes, buying",
            "Human Resources": "employment, payroll, leave, hiring",
            "General Inquiry": "anything else"
        }
        for row in ds_sup:
            if len([x for x in items if x["category"] == "app.support_triage"]) >= 400: break
            if row.get("language") != "en" or not row.get("body") or row.get("queue") not in sup_criteria: continue
            items.append({
                "category": "app.support_triage",
                "state": f"Subject: {row.get('subject', '')}\n\nBody: {str(row.get('body', ''))[:2000]}",
                "questions": {"queue": {"type": "choice", "instructions": "Which support queue should handle this ticket?", "criteria": sup_criteria}},
                "gold": {"queue": {"type": "choice", "label": row["queue"]}}
            })

        # 5. Email Spam
        ds_spam = load_dataset("SetFit/enron_spam", split="test")
        spam_crit = {"true": "unsolicited spam or promotional email", "false": "legitimate email communication"}
        for row in ds_spam:
            if len([x for x in items if x["category"] == "app.email_spam"]) >= 400: break
            is_sp = "true" if int(row.get("label", 0)) == 1 else "false"
            items.append({
                "category": "app.email_spam",
                "state": f"Subject: {row.get('subject', '')}\n\nMessage: {str(row.get('message', ''))[:2000]}",
                "questions": {"is_spam": {"type": "choice", "instructions": "Is this email unsolicited spam or bulk marketing?", "criteria": spam_crit}},
                "gold": {"is_spam": {"type": "choice", "label": is_sp}}
            })

        # 6. Phishing
        ds_phish = load_dataset("zefang-liu/phishing-email-dataset", split="train")
        phish_crit = {"true": "phishing, scam, or fraudulent email", "false": "legitimate safe email"}
        for row in ds_phish:
            if len([x for x in items if x["category"] == "app.phishing"]) >= 400: break
            is_ph = "true" if row.get("Email Type") == "Phishing Email" else "false"
            items.append({
                "category": "app.phishing",
                "state": f"Email: {str(row.get('Email Text', ''))[:2000]}",
                "questions": {"is_phishing": {"type": "choice", "instructions": "Is this email a phishing or scam attempt?", "criteria": phish_crit}},
                "gold": {"is_phishing": {"type": "choice", "label": is_ph}}
            })

        if items:
            ADAPTED_DIR.mkdir(parents=True, exist_ok=True)
            with open(local_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)

        return items

__all__ = ["LayaSource"]
