# Zadanie 3 – Tréning neurónových sietí v Dash aplikácii

Viacstránková webová aplikácia postavená na Plotly Dash, PyTorch a TensorBoarde. Dátové funkcie, modely, tréning a čítanie výsledkov sú oddelené v module `backend/`.

## Požiadavky

- Python 3.10 alebo novší
- Internetové pripojenie pri prvom načítaní datasetov MNIST/CIFAR alebo pri načítaní Heart Disease z URL
- Odporúčané je virtuálne prostredie; tréning CNN na CPU môže trvať dlhšie

## Inštalácia

V termináli otvorte priečinok `zadanie_3_dash` a spustite:

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Potom otvorte v prehliadači adresu `http://127.0.0.1:8050`.

> Inštalácia PyTorch/torchvision závisí od systému a dostupnosti GPU. Ak pip nevie vybrať kompatibilný balík, nainštalujte `torch` a `torchvision` podľa oficiálnych inštrukcií PyTorch a až potom znovu spustite `pip install -r requirements.txt`.

## Ako aplikáciu používať

1. Vyberte Heart Disease, MNIST, CIFAR-10, CIFAR-100 alebo vlastný CSV cez URL/súbor a stlačte **Načítať dáta**.
2. Po načítaní vyberte ANN alebo CNN. Heart Disease ponúka ANN; CNN je dostupná pre obrazové datasety.
3. Pri ANN zadajte počet skrytých vrstiev a zoznam neurónov, napríklad `128, 64`. Pri CNN vyberte EfficientNet-B0, MobileNetV3 Small alebo ResNet-18.
4. Potvrďte architektúru. Nastavte epochy, batch size, learning rate a pomer train/validation. Optimalizátor zostáva Adam.
5. Spustite tréning. Dokončené behy a ich grafy sa zobrazia na stránke **Výsledky**.

## Štruktúra projektu

```text
zadanie_3_dash/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── styles.css
├── pages/
│   ├── __init__.py
│   ├── training.py
│   └── results.py
├── backend/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── training.py
│   └── results.py
├── data/                 # stiahnuté dataset súbory
└── runs/                 # TensorBoard event súbory a metadata behov
```

## Použité požadované mechanizmy Dash

- **Dash Pages:** samostatné `/` a `/results`.
- **Dynamický layout:** sekcie modelu a tréningu sa zobrazia až po úspešnom predchádzajúcom kroku.
- **`prevent_initial_call=True`:** načítanie datasetu, potvrdenie architektúry a spustenie tréningu nereagujú pri prvom vykreslení stránky.
- **`running`:** pri načítaní/tréningu sa zobrazí stav a príslušné tlačidlo sa dočasne zablokuje.
- **`PreventUpdate` a `no_update`:** chýbajúce vstupy nespustia akciu; pri chybe načítania sa zachová posledný platný stav.
- **`ctx.triggered_id`:** callback datasetu rozlišuje načítanie dát od vyčistenia výberu.
- **TensorBoard:** každý beh má vlastný priečinok v `runs/`; výsledková stránka načíta metriky z event súborov cez `EventAccumulator`.

## Poznámky k dátam a tréningu

- Heart Disease načítava CSV z URL uvedenej v notebooku; cieľový stĺpec je `target`.
- Vlastný zdroj je CSV s cieľovým stĺpcom. Kategorické vstupné stĺpce sa zakódujú na číselné príznaky.
- Pri Heart Disease sa 15 % záznamov odloží na test. Zvyšok sa rozdelí medzi train a validation podľa ovládača.
- MNIST/CIFAR používajú oficiálnu testovaciu množinu. Ovládač rozdeľuje iba pôvodnú tréningovú množinu.
- ANN pri obrázkoch vstup najprv sploští na vektor. CNN zachováva rozmery obrazu a používa len `weights=None`.
- Normalizácia tabuľkových dát sa fituje iba na tréningovej časti, aby sa validačné ani testovacie dáta nepoužili pri príprave vstupu modelu.
- Modely sa učia od náhodne inicializovaných váh. Výsledky závisia od nastavení, hardvéru a náhodnosti tréningu.

## Užitočné príkazy

```bash
# spustenie Dash aplikácie
python app.py

# samostatný TensorBoard, ak ho chcete pozerať aj mimo výsledkovej stránky
tensorboard --logdir runs
```

V projekte sa pri importe backendu automaticky nenačítava dataset a nespúšťa tréning.
