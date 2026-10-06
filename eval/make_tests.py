"""Build a deterministic Clef test set: BANKING77 77-way, mixed-type multi-question, and long-context records."""
import csv, json, random, sys
src, out = sys.argv[1], sys.argv[2]
rng = random.Random(1234)
rows = list(csv.DictReader(open(src)))
labels = sorted({r["category"] for r in rows})
criteria77 = {l: l.replace("_", " ") for l in labels}
recs = []
for r in rng.sample(rows, 150):
    recs.append({"set": "banking77", "gold": {"intent": r["category"]}, "request": {"model": "clef", "state": r["text"],
        "questions": {"intent": {"type": "choice", "instructions": "Which banking intent does the customer message express?", "criteria": criteria77}}}})
for r in rng.sample(rows, 50):
    recs.append({"set": "mixed", "request": {"model": "clef", "state": {"channel": rng.choice(["chat", "email", "phone"]), "customer_tier": rng.choice(["free", "premium"]), "message": r["text"]},
        "questions": {
            "card_related": {"type": "noul", "instructions": "Is the message about a physical or virtual card?"},
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today", "Immediately"]},
            "team": {"type": "choice", "instructions": "Which team should handle it?", "criteria": {"cards": "Card issues", "payments": "Transfers and payments", "account": "Account and identity", "fx": "Exchange rates and currency"}},
            "angry": {"type": "noul", "instructions": "Does the customer sound frustrated?"}}}})
texts = [r["text"] for r in rows]
for target_words, needle_pos in ((6000, 0.1), (12000, 0.5), (24000, 0.9), (45000, 0.5)):
    filler = []
    while sum(len(t.split()) for t in filler) < target_words: filler.append(rng.choice(texts))
    i = int(len(filler) * needle_pos)
    filler.insert(i, "URGENT FROM COMPLIANCE: the customer's account was frozen due to suspected fraud.")
    recs.append({"set": "long", "request": {"model": "clef", "state": {"transcript": filler},
        "questions": {"frozen": {"type": "noul", "instructions": "Does the transcript say an account was frozen?"},
                      "topic": {"type": "choice", "instructions": "What is the most serious issue mentioned?", "criteria": {"fraud": "Fraud or account freeze", "card": "Card delivery", "fees": "Fees or charges"}}}}})
json.dump(recs, open(out, "w"))
print(len(recs), "records")
