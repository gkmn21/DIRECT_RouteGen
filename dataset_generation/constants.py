
TIME_CONSTRAINTS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

##
# OSM tags for tourist attractions and accommodations
##
TOURIST_ATTR_TAGS = [
    ('tourism', 'artwork'),
    ('tourism', 'attraction'),
    ('tourism', 'viewpoint'),
    ('tourism', 'museum'),
    ('tourism', 'gallery'),
    ('tourism', 'zoo'),
    ('tourism', 'theme_park'),
    ('tourism', 'aquarium'),
    ('tourism', 'wine_cellar'),
    ('tourism', 'highlight'), 
    ('tourism', 'tower_viewer'),
    ('tourism', 'arts_centre'),
    ('tourism', 'history'),
    ('tourism', 'Atelier und Künstlerhaus'),
    ('tourism', 'clock'),
    ('tourism', 'zeitgenössische_Kunst'),
    ('tourism', 'shopping'),
    ('tourism', 'exhibition'),
    ('tourism', 'yes')
]


CLEANED_CATEGORIES_AND_TAGS = {'Nature & Outdoor Recreation': [
  ('leisure', 'park'),
  ('leisure', 'garden'),
  ('leisure', 'nature_reserve'),
  ('natural', 'beach'),
  ('landuse', 'meadow'),
  ('landuse', 'forest'),
  ('landuse', 'orchard'),
  ('landuse', 'recreation_ground'),
  ('place', 'islet'),
  ('leisure', 'water_park'),
  ('leisure', 'trampoline_park'),
  ('tourism', 'viewpoint'),
  ('tourism', 'zoo'),
  ('tourism', 'aquarium')],
 'Family & Kids Attractions': [
('leisure', 'playground'),
  ('leisure', 'indoor_play'),
  ('playground', 'splash_pad')
],
 'Entertainment & Amusement': [('tourism', 'theme_park'),
  ('leisure', 'amusement_arcade'),
  ('leisure', 'escape_game'),
  ('leisure', 'bowling_alley'),
  ('leisure', 'ice_rink'),
  ('leisure', 'miniature_golf'),
  ('leisure', 'disc_golf_course'),
  ('sport', 'laser_tag'),
  ('sport', 'karting'),
  ('sport', 'roller_skating'),
  ('sport', 'paintball'),
  ('tourism', 'shopping')],
 'Culture & Arts': [('tourism', 'attraction'),
  ('tourism', 'artwork'),
  ('tourism', 'arts_centre'),
  ('tourism', 'Atelier und Künstlerhaus'),
  ('tourism', 'zeitgenössische_Kunst'),
  ('tourism', 'museum'),
  ('tourism', 'gallery'),
  ('leisure', 'music_venue'),
  ('tourism', 'wine_cellar'),
  ('tourism', 'exhibition')],
 'Water & Marina Activities': [('leisure', 'marina'),
  ('seamark:harbour:category', 'marina'),
  ('leisure', 'swimming_area'),
  ('harbour', 'yes'),
  ('water', 'lake')
    ],
 'Heritage & Landmarks': [
  ('historic', 'yes'),
  ('historic', 'heritage'),
  ('tourism', 'tower_viewer'),
  ('tourism', 'history'),
  ('tourism', 'clock')
  ]}



ACCOMMODATION_TAGS = [
    # Provide tags related to accommodations here eg. ('tourism', 'hotel')
]


NEW_NODE_CATEGORIES = [
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
