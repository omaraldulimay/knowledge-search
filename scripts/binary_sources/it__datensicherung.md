# Datensicherung und Wiederherstellung

## Ziel

Dieses Dokument beschreibt, wie die Elbkontor Logistik GmbH Daten sichert. Es richtet sich an die IT und an Datenverantwortliche.

## Sicherungsplan

Die ELKO-Datenbank wird stündlich gesichert und die Sicherungen 30 Tage aufbewahrt. Teamlaufwerke (K:, V:, L:) und Postfächer werden täglich um 22:00 Uhr inkrementell und samstags um 01:00 Uhr vollständig gesichert; diese Sicherungen werden 12 Wochen aufbewahrt. Laptops werden nicht gesichert, Arbeitsdateien gehören auf die Laufwerke.

## Speicherorte

Die Sicherungen liegen im Rechenzentrum Hammerbrook ("RZ-H") und eine Kopie wöchentlich auf Bändern im Tresor bei der Geschäftsführung in Gebäude A, Raum 3.02.

## Wiederherstellung

Für ELKO gilt ein Wiederherstellungsziel von 4 Stunden (RTO) und höchstens 1 Stunde Datenverlust (RPO). Wiederherstellungstests laufen vierteljährlich; der letzte Test am 06.10.2026 stellte die ELKO-Datenbank in 2 Stunden 50 Minuten wieder her.

## Zuständigkeit

Verantwortlich ist die Teamleitung IT, Ole Brandt. Wiederherstellungen einzelner Dateien beantragen Beschäftigte im Lotse (Kategorie Sonstiges), Bearbeitung binnen eines Arbeitstags. Ein Ausfall von ELKO ist P1, siehe IT-Support und Tickets.
