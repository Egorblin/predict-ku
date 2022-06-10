import streamlit as st
import pandas as pd
import numpy as np
import pickle

with st.echo(code_location='below'):
    @st.cache(allow_output_mutation=True)
    def get_data():
        types = pd.read_csv('types.csv')
        all_models = pickle.load(open('models', 'rb'))
        return types, all_models

    def predict_KU(X, type_1, type_2, models_configs):
        X = X[(X['Вид учреждения'] == type_1) & (X['Тип учреждения'] == type_2)]
        X_small = X.drop(columns=['Год', 'Вид учреждения', 'Тип учреждения', 'Номер месяца'])
        models_type = all_models[f'{type_2}_{type_1}'][0]
        
        y_pred = np.array([0] * X.shape[0], dtype=np.float64)
        n = 0
        for model_config in models_type.keys():
            config = models_type[model_config]
            model = config[0]
            enc = config[1]
            sc = config[2]
            
            X_pred = enc.transform(X_small)
            X_pred = sc.transform(X_pred)
            
            y_pred += model.predict(X_pred)
            n += 1
        
        def_ = lambda x: x if x > 0 else 0
        y_pred = np.array([def_(x) for x in y_pred])
        y_pred = np.round(y_pred / n)
        
        y_pred = pd.DataFrame(data={'Предсказанное количество исследований с КУ': y_pred}, index=X.index)
        X = pd.concat([X, y_pred], axis=1)
        
        for j in X.index:
            if X.loc[j, 'Количество исследований'] < X.loc[j, 'Предсказанное количество исследований с КУ']:
                X.loc[j, 'Предсказанное количество исследований с КУ'] = \
                X.loc[j, 'Количество исследований']
        if type_1 == 'Взрослое' and type_2 == 'Стационарное':
            X.loc[:, 'Погрешность'] = 50 + 0.2 * X.loc[:, 'Предсказанное количество исследований с КУ']
        elif type_1 == 'Детское' and type_2 == 'Стационарное':
            X.loc[:, 'Погрешность'] = 5 + 0.05 * X.loc[:, 'Предсказанное количество исследований с КУ']
        return X

    '''
    Прогнозирование количества исследований с контрастным усилением
    '''

    '''
    Инструкция:
    1. Выбрать медицинское учреждение, для которого будет сделан прогноз
    2. Выбрать номер месяца или месяцев для прогнозирования
    3. Ввести планируемые значения величин для предстоящего месяца или месяцев
    4. Проверить верно ли введены данные в таблицу
    5. Получить результат
    '''

    types, all_models = get_data()

    #st.write(types.head())

    med = st.selectbox('Медицинское учреждение', sorted(types['МО']), key='med')

    year = 2022
    type_1 = types[types['МО'] == med].iloc[0][ 'Вид учреждения']
    type_2 = types[types['МО'] == med].iloc[0][ 'Тип учреждения']
    n_device = types[types['МО'] == med].iloc[0]['Количество аппаратов']

    month_vals = [2, 3]
    month = st.multiselect('Номер месяца для прогнозирования', month_vals, default=month_vals[0], key='month')
    month.sort()

    researchs = [] 
    days = []
    shifts = []
    

    for m in month:
        researchs.append(st.number_input(f'Введите планируемое количество исследований в {m} месяце 2022 года', 
                                        min_value=0, 
                                        max_value=None, 
                                        step=1, 
                                        key='research')
                        )

        days.append(st.number_input(f'Введите планируемое количество отработанных дней в {m} месяце 2022 года', 
                                        min_value=0, 
                                        max_value=None, 
                                        step=1, 
                                        key='days')
                        )

        shifts.append(st.number_input(f'Введите планируемое количество отработанных смен в {m} месяце 2022 года', 
                                        min_value=0, 
                                        max_value=None, 
                                        step=1, 
                                        key='shift')
                        )

    len_X = len(month)
    med = [med] * len_X
    devices = [n_device] * len_X
    year = [year] * len_X
    type_1 = [type_1] * len_X
    type_2 = [type_2] * len_X
    all_month = []
    for m in month:
        all_month.append((year[0]-2015) * 12 + m)

    d = {'Год': year, 'МО': med, 'Вид учреждения': type_1, 'Тип учреждения': type_2, 'Номер месяца': month, 'Количество исследований': researchs, \
        'Количество отработанных дней': days, 'Количество отработанных смен': shifts, 'Количество аппаратов': devices, 'Все месяцы': all_month}


    '''
    Данные, на основе которых будет производиться прогноз
    '''
    X = pd.DataFrame(data=d)
    st.write(X.drop(columns='Все месяцы'))

    '''
    Результат
    '''
    X_pred = predict_KU(X, type_1[0], type_2[0], all_models)
    st.write(X_pred.loc[:, ['Год', 'Номер месяца', 'МО', 'Предсказанное количество исследований с КУ', 'Погрешность']])