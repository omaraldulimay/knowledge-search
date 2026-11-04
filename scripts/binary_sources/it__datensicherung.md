# Datensicherung und Wiederherstellung

## Ziel

Dieses Dokument beschreibt, wie die Elbkontor Logistik GmbH ihre Daten sichert und im Notfall wiederherstellt. Es richtet sich an IT-Mitarbeitende und an Datenverantwortliche der Abteilungen.

## Was wird gesichert

Gesichert werden die Teamlaufwerke, die Postfächer, die Datenbank des Lagerverwaltungssystems, das ERP-System und die Konfiguration der Netzwerkgeräte. Lokale Daten auf Laptops werden nicht gesichert. Mitarbeitende müssen Arbeitsdateien daher auf den Teamlaufwerken oder in der Cloud-Ablage ablegen.

## Sicherungsplan

Die Datenbank des Lagerverwaltungssystems wird stündlich inkrementell und nachts vollständig gesichert. Dateiserver werden täglich um 22:00 Uhr gesichert. Postfächer werden täglich gesichert und 30 Tage vorgehalten. Wöchentliche Vollsicherungen werden 12 Wochen aufbewahrt, monatliche Sicherungen 12 Monate und Jahressicherungen zehn Jahre, soweit steuerrechtliche Fristen gelten.

## Drei-Kopien-Regel

Von jedem Datenbestand existieren drei Kopien: das Produktivsystem, eine Sicherung im zweiten Rechenzentrum in Hamburg-Harburg und eine verschlüsselte, unveränderliche Kopie bei einem externen Anbieter in Deutschland. Die Verschlüsselung erfolgt mit AES-256, die Schlüssel liegen getrennt in einem Tresor der IT-Leitung.

## Wiederherstellungsziele

Für das Lagerverwaltungssystem gilt eine maximale Wiederherstellungszeit von 4 Stunden und ein maximaler Datenverlust von 1 Stunde. Für das ERP-System sind es 8 Stunden und 24 Stunden Datenverlust. Für Dateiserver gilt 24 Stunden. Die Ziele werden jährlich überprüft.

## Test der Wiederherstellung

Einmal pro Quartal wird die Wiederherstellung einer zufälligen Sicherung getestet und dokumentiert. Einmal im Jahr findet eine Notfallübung statt, in der das Lagerverwaltungssystem im zweiten Rechenzentrum neu gestartet wird. Ergebnisse gehen an die Geschäftsführung.

## Wiederherstellung einzelner Dateien

Versehentlich gelöschte Dateien können vom IT-Support aus der Sicherung zurückgeholt werden, wenn sie in den letzten 90 Tagen gelöscht wurden. Erstellen Sie dazu ein Ticket mit Pfad und ungefährem Zeitpunkt der Löschung.

## Verantwortlichkeiten

Für den Betrieb der Sicherung ist der Teamleiter Infrastruktur, Herr Piet Lohmeyer, verantwortlich. Er berichtet quartalsweise an die Geschäftsführung. Abweichungen, etwa fehlgeschlagene Sicherungen, lösen automatisch ein Ticket der Priorität 2 aus und müssen binnen eines Arbeitstages behoben werden.
