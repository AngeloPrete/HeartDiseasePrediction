import pandas as pd
from pyswip import Prolog

def aggiungi_fatti(data,prolog):
    # Aggiungi i fatti
    for i, row in data.iterrows():
        colesterolo = row['Cholesterol']
        eta = row['Age']
        pressione_sanguigna = row['RestingBP']
        sesso = row['Sex']
        chest_pain = row['ChestPainType']
        glicemia_alta = row['FastingBS']
        result_ecg = row['RestingECG']
        maxHR = row['MaxHR']
        exang = row['ExerciseAngina']
        oldpeak = row['Oldpeak']
        slope = row['ST_Slope']
        target = row['HeartDisease']

        prolog.assertz(f"paziente({i+1})")
        prolog.assertz(f"colesterolo({i+1}, {colesterolo})")
        prolog.assertz(f"eta({i+1}, {eta})")
        prolog.assertz(f"pressione_sanguigna({i+1}, {pressione_sanguigna})")
        prolog.assertz(f"sesso({i+1}, {sesso})")
        prolog.assertz(f"chest_pain({i+1}, {chest_pain})")
        prolog.assertz(f"glicemia_alta({i+1}, {glicemia_alta})")
        prolog.assertz(f"result_ecg({i+1}, {result_ecg})")
        prolog.assertz(f"fc_max({i+1}, {maxHR})")
        prolog.assertz(f"exang({i+1}, {exang})")
        prolog.assertz(f"oldpeak({i+1}, {oldpeak})")
        prolog.assertz(f"slope({i+1}, {slope})")
        prolog.assertz(f"presenza_malattia({i+1}, {target})")

def aggiungi_regole(prolog):
    # Aggiungo regole per inferenza
    prolog.assertz("colesterolo_alto(ID) :- colesterolo(ID, Col), Col > 250")
    prolog.assertz("colesterolo_moderato(ID) :- colesterolo(ID, Col), Col >= 200, Col =< 250")
    prolog.assertz("colesterolo_basso(ID) :- colesterolo(ID, Col), Col < 200")

    prolog.assertz("pressione_sanguigna_alta(ID) :- pressione_sanguigna(ID, Bps), Bps > 140")
    prolog.assertz("pressione_sanguigna_moderata(ID) :- pressione_sanguigna(ID, Bps), Bps >= 120, Bps =< 140")
    prolog.assertz("pressione_sanguigna_bassa(ID) :- pressione_sanguigna(ID, Bps), Bps < 120")

    prolog.assertz("oldpeak_alto(ID) :- oldpeak(ID, Millimetri), Millimetri > 2")
    prolog.assertz("oldpeak_moderato(ID) :- oldpeak(ID, Millimetri), Millimetri >= 1, Millimetri =< 2")
    prolog.assertz("oldpeak_basso(ID) :- oldpeak(ID, Millimetri), Millimetri < 1")
    
def get_categoria_pressione_sanguigna(id,prolog):
    
    result = list(prolog.query(f"pressione_sanguigna_bassa({id})"))

    if(result):
        return 'Normal'
    else :
        result = list(prolog.query(f"pressione_sanguigna_moderata({id})"))
        if(result):
            return 'Moderate'
        else :
            return 'High'

def get_categoria_colesterolo(id,prolog):
    
    result = list(prolog.query(f"colesterolo_basso({id})"))

    if(result):
        return 'Normal'
    else :
        result = list(prolog.query(f"colesterolo_moderato({id})"))
        if(result):
            return 'Moderate'
        else :
            return 'High'

def get_categoria_oldpeak(id,prolog):
    
    result = list(prolog.query(f"oldpeak_basso({id})"))

    if(result):
        return 'Normal'
    else :
        result = list(prolog.query(f"oldpeak_moderato({id})"))
        if(result):
            return 'Moderate'
        else :
            return 'High'
        
def inferisci_feature(data,prolog):

    for id, row in data.iterrows():
        
        row_id = id[0] if isinstance(id, tuple) else id  # Assicura che id sia un intero
        data.at[row_id, 'RestingBP_categoria'] = get_categoria_pressione_sanguigna(row_id + 1,prolog)
        data.at[row_id, 'Cholesterol_categoria'] = get_categoria_colesterolo(row_id + 1,prolog)
        data.at[row_id, 'Oldpeak_categoria'] = get_categoria_oldpeak(row_id + 1,prolog)
