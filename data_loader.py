import json

def load_candidates(filepath="dummy-resume.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        candidate_list = json.load(f)

    phone_index = {}
    for candidate in candidate_list:
        phone = candidate.get("phone")
        if phone:
            phone_index[phone] = candidate

    return phone_index

def get_candidate_by_phone(phone_number, db):
    return db.get(phone_number)