FINAL_CATEGORIES = [
    'junction', # category for non-poi nodes in network
    'accommodation',
    'food establishment',
    'cafe',
    'Nature & Outdoor Recreation',
    'Family & Kids Attractions',
    'Entertainment & Amusement',
    'Culture & Arts',
    'Water & Marina Activities',
    'Heritage & Landmarks',
    'other'
]


NEW_VISIT_DURATION_BASED_ON_CATEGORIES =  { # in seconds
    'junction': [0, 0],
    'accommodation': [0, 0],
    'food establishment': [1*3600, 2*3600],
    'cafe': [1*3600, 2*3600],
    'Nature & Outdoor Recreation': [1*3600, 2*3600],
    'Family & Kids Attractions': [1*3600, 2*3600],
    'Entertainment & Amusement': [1*3600, 3*3600],
    'Culture & Arts': [1 * 3600, 2 * 3600],
    'Water & Marina Activities': [1 * 3600, 3 * 3600],
    'Heritage & Landmarks': [0.5 * 3600, 1 * 3600],
    'other': [0.5*3600, 2*3600]
}

VERONA_DATASET_CATEGORIES = [
    'Musei e Centri Espositivi',
    'Monumenti',
    'Chiese'
]