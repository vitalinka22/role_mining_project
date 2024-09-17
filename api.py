import requests
from requests_negotiate_sspi import HttpNegotiateAuth
from collections import defaultdict
import pickle
import csv
import pandas as pd
from pathlib import Path
from urllib.parse import quote


def fetch_user_permissions(save_path, group):
    # Kodieren des Gruppennamens
    group_encoded = quote(group)

    # Beispiel-URL (anonymisiert)
    uri = f"https://example-api.com/api/v1/groups/{group_encoded}/members?recursive=true"

    # Ausgabe der URL zur Überprüfung
    print(f"Aufgerufene URL: {uri}")

    # Setze den Access-Token (falls benötigt)
    headers = {
        'Content-Type': 'application/json'
    }
    members = []  # Initialisiere members als leere Liste
    try:
        response = requests.get(uri, headers=headers, auth=HttpNegotiateAuth())
        response.raise_for_status()
        members = response.json()
        # Gruppenmitglieder konnten erfolgreich abgerufen werden und sind im Array members gespeichert.
        print(f"Anzahl der Gruppenmitglieder: {len(members)}")
    except requests.exceptions.RequestException as e:
        # Ausgabe des Fehlers (für Logging..)
        print(f"Fehler: {e}")

    users = []
    objectGuids = []
    for block in members:
        displayName = block.get("displayName")
        objectGuid = block.get("objectGuid")
        if block.get("givenName"):  # Hier kannst du die Bedingung nach Bedarf anpassen
            users.append(displayName)
            objectGuids.append(objectGuid)

    print(users)

    user_sets = defaultdict(set)

    for user, objectGuid in zip(users, objectGuids):
        # Beispiel-URL für den aktuellen Benutzer (anonymisiert)
        url = f"https://example-api.com/api/v1/users/{objectGuid}/groups?recursive=true"

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            # API-Anfrage senden
            response = requests.get(url, headers=headers, auth=HttpNegotiateAuth())
            response.raise_for_status()  # Fehler auslösen, wenn der Statuscode nicht 2xx ist

            groups = response.json()

            # Erfolgreiche Antwort und Gruppendaten
            print(f"Anzahl der Gruppen für {user}: {len(groups)}")

        except requests.exceptions.RequestException as e:
            # Fehlerbehandlung
            print(f"Fehler bei {user}: {e}")
            continue  # Überspringe den aktuellen Benutzer bei Fehler

        # Speichere die samAccountNames der Gruppen in user_sets
        if groups:  # Überprüfen, ob die Antwort Daten enthält
            for group in groups:
                sam_account_name = group.get("samAccountName")
                if sam_account_name:
                    user_sets[user].add(sam_account_name)

    # Ausgabe der Ergebnisse
    for user, groups in user_sets.items():
        print(f"Benutzer: {user}, Gruppen: {groups}")
        print("____")

    # Zusammenführen aller Sets
    combined_set = set()
    for user_set in user_sets.values():
        combined_set.update(user_set)

    # Sortierte Liste aller eindeutigen samAccountNames
    combined_array = sorted(combined_set)
    print(len(combined_array))

    # Speichern der sortierten Liste in einer Datei
    # with open("list_of_permissions.txt", "wb") as file:
    #    pickle.dump(combined_array, file)

    # Darstellung den Sets von den Benutzern in den binärischen Form
    user_sets_bin = {user_name: [] for user_name in user_sets}
    for user_name, user_set in user_sets.items():
        user_sets_bin[user_name] = [1 if x in user_set else 0 for x in combined_array]

    # Benutzerrollen und den entsprechenden Benutzer in eine CSV-Datei schreiben
    csv_filename = "user_permissions.csv"
    users = []
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for user_name, user_set in user_sets_bin.items():
            writer.writerow([user_name] + user_set)
            users.append(user_name)

    # Benutzerliste speichern
    with open("list_of_users.txt", "wb") as file:
        pickle.dump(users, file)

    # Alle Benutzerrollen speichern
    df = pd.DataFrame(combined_array)
    print(len(combined_array))
    df.to_csv("data_permissions.csv", index=False, header=False)

    base_path = Path(__file__).parent
    permissions_list_path = base_path / "data_permissions.csv"
    df = pd.read_csv(permissions_list_path, header=None, encoding='latin1')

    # Umwandeln des DataFrames in eine Liste (jede Zeile wird ein Element)
    list_of_permissions = df.values.flatten().tolist()

    # Ausgabe der Liste
    print("---")
    print(len(list_of_permissions))

