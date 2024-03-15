import fastf1 as f1
import pandas as pd


def get_session(year, round, session_type):
    session_data = f1.get_session(year, round, session_type)
    session_data.load(laps=False, telemetry=False, weather=False, messages=False)

    session_results = get_session_results(session_data.results)
    return session_results

    session = {
        'results': get_session_results(session_data.results),
        #'session_info': session_data.session_info,
        #'weather_data': session_data.weather_data
    }

    return session


def get_session_results(session_results):
    session_prep = pd.DataFrame(session_results, columns=['DriverNumber', 'Abbreviation', 'GridPosition', 'Position', 'ClassifiedPosition', 'Status', 'Time', 'Points'])
    #session_cols_int = ['DriverNumber', 'GridPosition', 'ClassifiedPosition', 'Position', 'Points']
    #session_cols_int = ['DriverNumber', 'GridPosition', 'Position', 'Points']
    #session_prep[session_cols_int] = session_prep[session_cols_int].astype('Int64')

    session_new = pd.DataFrame()
    session_new['Abbreviation'] = session_prep['Abbreviation']
    session_new['GridPosition'] = session_prep['GridPosition']
    session_new['Finished'] = session_prep.apply(lambda x: 1.0 if pd.to_numeric(x['ClassifiedPosition'], errors='coerce') == x['Position'] else 0.0, axis=1)
    session_new['Position'] = session_prep['Position']

    return session_new


def get_session_drivers(session_results):
    drivers = pd.unique(session_results['Abbreviation'])
    return drivers


def get_sessions(year):
    num_sessions = len(f1.get_event_schedule(year, include_testing=False).values)
    
    sessions = []
    for i in range(0, num_sessions):
        session = get_session(year, i+1, 'R')
        sessions.append(session)

    return sessions