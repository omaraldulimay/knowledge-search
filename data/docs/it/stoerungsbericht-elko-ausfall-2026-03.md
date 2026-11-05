# Störungsbericht: ELKO-Ausfall am 29.03.2026

## Ablauf

Am Sonntag, 29.03.2026, startete um 02:00 Uhr das Release 4.2. Die Datenbank-Indexmigration blockierte, ELKO war ab 03:15 Uhr nicht erreichbar. Ab 05:30 Uhr begann die Frühschicht in Halle 2 und Halle 3 ohne System. Die Störung dauerte insgesamt 4 Stunden 20 Minuten, ELKO lief wieder um 07:35 Uhr nach Rollback auf Release 4.1.

## Auswirkungen

Elf Lkw hatten durchschnittlich 70 Minuten Verspätung. Drei Anlieferungen von Nordhafen Frische an Tor 7 wurden manuell auf Papier erfasst und danach nachgebucht.

## Ursache

Die Indexmigration wurde nur gegen einen Testdatenbestand von 5 Prozent geprüft. Die Rollback-Entscheidung wurde erst nach 130 Minuten getroffen.

## Beschlossene Maßnahmen

1. Rollback-Regel: nach 60 Minuten ohne stabiles System wird zurückgerollt (Entscheidung Imke Ohlsen am 02.04.2026).
2. Go/No-Go-Prüfung am Dienstag vor dem Release.
3. Test mit vollständiger Datenkopie.
4. Papier-Notfallset "Notfallmappe ELKO" an Tor 7 und Tor 5.

## Gelernt

Die IT-Bereitschaft wurde zu spät informiert; seitdem ruft der Leitstand Halle 2 bei ELKO-Ausfall sofort Durchwahl 3077 beziehungsweise 3070.
