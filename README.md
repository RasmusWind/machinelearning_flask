# machinelearning_flask

Denne opgave handler om at kunne træne og gentræne en model til at genkende enkelte tal.
Ved hjælp af en simpel frontend, lavet med Flask, kan man tegne et tal, derefter gætter programmet og spørger efter om det gæt var korrekt, derefter gentræner den modellen.

Datasættet kommer fra keras mnist som er et datasæt der indeholder arrays af pixel værdier som repræsentere et billede af et tal.

Modellen bliver trænet "supervised", da den bliver givet det rigtige svar når vi gentræner den. Grunden til at jeg bruger supervised learning er fordi det er en simpel model, og at "supervised learning" er en simpel form for læring. 

Den nuværende måde at problemet bliver løst på kræver et svar per tal tegnet, så det er langt fra den mest optimale løsning. 
Problemet er ikke komplekst da det bare er simple tal der er tale om.
Modellen forbedres hver gang den får et svar og en tegning, så den bliver allerede forbedret.
