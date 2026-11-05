# Sample data: Elbkontor Logistik GmbH

Fictional mid-sized logistics company in Hamburg-Billbrook. All names, numbers, extensions and rules are
invented. The point of the corpus: **every answerable question needs knowledge that exists only inside
this company** (own systems such as ELKO, Lotse and Zeitanker, status codes, extensions, cost centres,
customer contracts, incident reports, meeting decisions), so no public chatbot can answer it.

52 German documents in `data/docs/<department>/` (departments `alle`, `hr`, `it`, `vertrieb`, `logistik`):
47 Markdown files, 3 PDFs and 2 DOCX files. The PDFs/DOCX are generated from
`scripts/binary_sources/*.md` by `python scripts/make_binary_docs.py` (reproducible).
`hr/gehaltsbaender.md` (salary bands, allowances, Elbstern bonus) is confidential: hr only, like every file in `hr/`.

| Department | Documents |
|---|---|
| alle (11) | reisekostenrichtlinie (+ 2023 version), urlaubsantrag, onboarding, arbeitszeit-homeoffice (+ 2024 version), brandschutz-notfall, betriebsrestaurant, verhaltenskodex, standortplan-durchwahlen, datenschutz-grundlagen (pdf) |
| hr (7) | gehaltsbaender, kuendigungsfristen, bewerbungsprozess, weiterbildung, betriebliche-altersvorsorge, elternzeit, probezeit-feedback (docx) |
| it (11) | vpn-einrichtung, passwortrichtlinie (+ 2022 version), mfa, hardware-bestellung, it-support-tickets, software-freigabe, bereitschaftsdienst, elko-betrieb, stoerungsbericht-elko-ausfall-2026-03, datensicherung (pdf) |
| vertrieb (10) | rabattstaffel-grosskunden (+ 2024 version), angebotsprozess, vertrieb-reisekosten, crm-pflege, provisionsmodell, kundenbewirtung, kundenakte-nordhafen-frische, kunden-sla-uebersicht, vertragsvorlagen (docx) |
| logistik (13) | elko-statuscodes, eskalationsplan-kuehlkette, stoerungsbericht-tor7-kuehlkette-2026-02, schichtmodelle, kundenvorgaben-annahmeschluss, protokoll-hallenleitung-2026-09, vertretung-bereitschaft, lagerordnung, gefahrgut-handling, fahrzeugpruefung, reklamation-transportschaden, ladungssicherung, zollabwicklung (pdf) |

## Versions

Four policies exist twice: the current version and an older one marked `superseded: true` with the same
`policy` key (reisekostenrichtlinie, arbeitszeit-homeoffice, passwortrichtlinie, rabattstaffel-grosskunden).
The newest-version rule must pick the current one (hotel 110 instead of 95 euro, 2 instead of 3 home-office
days, 14 instead of 10 characters, ...). The current documents say "ersetzt Fassung vom ...".

## Cross-references

Documents point at each other ("siehe Eskalationsplan Kühlkette"); the link index turns that into graph edges.
Some questions need two documents, for example:

| Question | Documents | Answer |
|---|---|---|
| Wer gibt meine Vertriebsreise für 1.800 Euro frei und wie viel darf das Hotel kosten? | alle/reisekostenrichtlinie + vertrieb/vertrieb-reisekosten | Bereichsleitung (Tjark Rohwedder); 140 Euro statt 110 Euro |
| Was tun bei unterbrochener Kühlkette an Tor 7 und was wurde nach dem Februar-Vorfall geändert? | logistik/eskalationsplan-kuehlkette + stoerungsbericht-tor7-kuehlkette-2026-02 | K3/S1, Durchwahl 5299; Lichtschranken alle 6 Monate warten |
| Wer gibt meinen Urlaub frei, wenn mein Teamleiter in Halle 2 krank ist? | alle/urlaubsantrag + logistik/vertretung-bereitschaft | die Hallenleitung, in Halle 2 Wiebke Thiele |
| Gilt der Großkundenrabatt auch für die Zollpauschale? | logistik/zollabwicklung + vertrieb/rabattstaffel-grosskunden | nein, 45 Euro je Anmeldung, nicht rabattfähig |
| Ich gehe neun Monate in Elternzeit: wie viel von 30 Urlaubstagen wird gekürzt? | alle/urlaubsantrag + hr/elternzeit | 9/12, also 22,5 Tage |
