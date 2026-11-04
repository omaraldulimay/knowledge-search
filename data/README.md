# Sample data: Elbkontor Logistik GmbH

Fictional mid-sized logistics company in Hamburg. All names, numbers and rules are invented.
35 German documents in `data/docs/<department>/` (departments `alle`, `hr`, `it`, `vertrieb`, `logistik`):
30 Markdown files, 3 PDFs and 2 DOCX files. The PDFs/DOCX are generated from
`scripts/binary_sources/*.md` by `python scripts/make_binary_docs.py` (reproducible).
`hr/gehaltsbaender.md` is confidential (hr only, like every file in `hr/`).

| Department | Documents |
|---|---|
| alle (8) | reisekostenrichtlinie, urlaubsantrag, onboarding, verhaltenskodex, arbeitszeit-homeoffice, brandschutz-notfall, betriebsrestaurant, datenschutz-grundlagen (pdf) |
| hr (7) | gehaltsbaender, kuendigungsfristen, bewerbungsprozess, weiterbildung, betriebliche-altersvorsorge, elternzeit, probezeit-feedback (docx) |
| it (7) | vpn-einrichtung, passwortrichtlinie, mfa, hardware-bestellung, it-support-tickets, software-freigabe, datensicherung (pdf) |
| vertrieb (7) | rabattstaffel-grosskunden, angebotsprozess, vertrieb-reisekosten, crm-pflege, provisionsmodell, kundenbewirtung, vertragsvorlagen (docx) |
| logistik (6) | gefahrgut-handling, lagerordnung, fahrzeugpruefung, reklamation-transportschaden, ladungssicherung, zollabwicklung (pdf) |

## Questions that need two documents

| # | Question | Documents to combine | Answer |
|---|---|---|---|
| 1 | Wie viel darf ein Hotel im Vertrieb pro Nacht kosten, und wie schnell muss ich abrechnen? | alle/reisekostenrichtlinie + vertrieb/vertrieb-reisekosten | 140 Euro statt 110 Euro; Abrechnung in 7 statt 14 Tagen |
| 2 | Welche Kündigungsfrist gilt für mich als Vertriebsmitarbeiter mit sechs Jahren im Unternehmen, wenn die Firma kündigt? | hr/kuendigungsfristen (+ hr/probezeit-feedback for the probation case) | zwei Monate zum Monatsende (ab 5 Jahren); in der Probezeit zwei Wochen |
| 3 | Wie arbeite ich im Homeoffice auf interne Systeme zu? | alle/arbeitszeit-homeoffice + it/vpn-einrichtung | bis zu 3 Tage pro Woche, VPN "Elbtunnel" mit MFA-Code |
| 4 | Was kostet ein Gefahrguttransport für einen Großkunden mit 10 Prozent Rabatt? | vertrieb/rabattstaffel-grosskunden + logistik/gefahrgut-handling | 18 Prozent Gefahrgutzuschlag, nicht rabattierbar |
| 5 | Wie viel darf ich Kunden auf einer Dienstreise zum Essen einladen? | vertrieb/kundenbewirtung + vertrieb/vertrieb-reisekosten | 60 Euro pro Person, Bewirtungsbeleg mit Anlass |
| 6 | Darf ich ein neues Programm auf meinem bestellten Laptop installieren? | it/hardware-bestellung + it/software-freigabe | nur freigegebene Software; sonst Ticket "Software-Freigabe" (ca. 10 Arbeitstage) |
| 7 | Wie viel Urlaub bekomme ich, wenn ich neun Monate in Elternzeit bin? | alle/urlaubsantrag + hr/elternzeit | 30 Tage, pro vollem Elternzeit-Monat ein Zwölftel kürzbar (9/12 gekürzt) |
| 8 | Ich habe mein Passwort und mein Handy verloren - wie komme ich wieder rein? | it/passwortrichtlinie + it/mfa | Passwort-Reset braucht den zweiten Faktor; MFA-Reset über IT-Support per Rückruf |
| 9 | Gilt der Großkundenrabatt auch für die Zollabfertigung? | logistik/zollabwicklung + vertrieb/rabattstaffel-grosskunden | nein, 45 Euro Pauschale pro Anmeldung, nicht rabattfähig |
| 10 | Wer bezahlt den ADR-Schein und zählt er auf mein Bildungsbudget? | hr/weiterbildung + logistik/gefahrgut-handling | das Unternehmen zahlt, kein Anrechnung auf die 1.200 Euro |
| 11 | Wie viel Zulage bekommt der Gefahrgutbeauftragte? | hr/gehaltsbaender + logistik/gefahrgut-handling | 150 Euro pro Monat (hr-only document: nur sichtbar für hr) |
