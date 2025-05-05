# Look AI – Probare virtuală a hainelor

Această aplicație folosește modelul CP-VTON și Intel OpenVINO pentru a genera o simulare realistă a hainelor asupra unei persoane.

## Cum se folosește
1. Pornește serverul:
```
uvicorn app.main:app --reload
```
2. Accesează în browser: [http://localhost:8000/web](http://localhost:8000/web)

3. Încarcă o imagine cu o persoană și o haină.

## Cerințe
- Python 3.8+
- Intel OpenVINO
