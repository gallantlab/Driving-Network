from enum import Enum, auto

from . import utils

class Participants(Enum):
    P1 = auto()
    P2 = auto()
    P3 = auto()
    P4 = auto()
    P5 = auto()
    P6 = auto()

Transform = 'driving'

ModelCategories = ['Fixation',
                   'Low-level Vision',
                   'High-level Vision',
                   'Motor',
                   'Motor Plans',
                   'Route progression',
                   'Goal-related',
                   'Path Integration',
                   'Cognitive Map']

ModelCategoryMembers = \
{	'Fixation':
        [
         'Eyetracking',
         'Gaze Grid',
        ],
    'Low-level Vision':
        [
         'Framewise Motion-Energy',
         'Retinotopic Motion-Energy',
        ],
    'High-level Vision':
        [
         'Gaze Semantics',
         'Frame Semantics',
         'Spatial Semantics',
         'Depth',
         'Scene Structure',
         'Affordance',
        ],
    'Motor':
        [
         'Controls'
        ],
    'Motor Plans':
        [
         'Turns Space Phase',
         'Turns Time Phase',
         'Future Path',
        ],
    'Route progression':
        [
         'Route Time Phase',
         'Route Space Phase',
         'Route Space Binned',
         'Route Time Binned',
         'Path Distance Remaining',
         'Beeline Distance Remaining',
        ],
    'Goal-related':
        [
         'Destination Grid Representation',
         'Destination Vector Log',
         'Destination-anchored Vector',
        ],
    'Path Integration':
        [
         'Beeline Distance Elapsed',
         'Path Distance Elapsed',
         'Time Elapsed',
         'Time Elapsed Phase',
         'Path Integration Egocentric',
         'Path Integration Allocentric',
        ],
    'Cognitive Map':
        [
         'Vehicles Spatial',
         'Pedestrians Spatial',
         'Road Graph',
         'Gaze Direction 6-fold',
         'Gaze Direction Binned',
         'Head Direction 6-fold',
         'Head Direction Binned',
         'Grid Cells',
         'Place Cells',
        ],

}

ModelCategoryIndices = \
{
    'Fixation': 			(0, 2),
    'Low-level Vision':		(2, 4),
    'High-level Vision':	(4, 10),
    'Motor':				(10, 11),
    'Motor Plans':			(11, 14),
    'Route progression':	(14, 20),
    'Goal-related':			(20, 23),
    'Path Integration':		(23, 29),
    'Cognitive Map':		(29, 38),
}

FeatureSpaces = ['Eyetracking',
                 'Gaze Grid',

                 'Framewise Motion-Energy',
                 'Retinotopic Motion-Energy',

                 'Gaze Semantics',
                 'Frame Semantics',
                 'Spatial Semantics',
                 'Depth',
                 'Scene Structure',
                 'Affordance',

                 'Controls',

                 'Turns Space Phase',
                 'Turns Time Phase',
                 'Future Path',

                 'Route Time Phase',
                 'Route Space Phase',
                 'Route Space Binned',
                 'Route Time Binned',
                 'Path Distance Remaining',
                 'Beeline Distance Remaining',

                 'Destination Grid Representation',
                 'Destination Vector Log',
                 'Destination-anchored Vector',

                 'Beeline Distance Elapsed',
                 'Path Distance Elapsed',
                 'Time Elapsed',
                 'Time Elapsed Phase',
                 'Path Integration Egocentric',
                 'Path Integration Allocentric',

                 'Vehicles Spatial',
                 'Pedestrians Spatial',
                 'Road Graph',
                 'Gaze Direction 6-fold',
                 'Gaze Direction Binned',
                 'Head Direction 6-fold',
                 'Head Direction Binned',
                 'Grid Cells',
                 'Place Cells',
                 ]


ROIs = ['RSC-pPrCu',
        'dPrCu',
        'aPrCu',
        'pmPFC',
        'OPA-pLPC',
        'aLPC',
        'pSFS',
        'IFS',
        'PPA-ColS',
        'aIns',
        'FP']

NavIndicesForNPI = []
others = list(range(14))
for category in ModelCategories[5:]:
    for model in ModelCategoryMembers[category]:
        index = utils.get_index(model, FeatureSpaces)
        if 'Gaze' not in model: # skip gaze directions, because this model is still too correlated with saccades and vision
            NavIndicesForNPI.append(index)
        else:
            others.append(utils.get_index(model, FeatureSpaces))