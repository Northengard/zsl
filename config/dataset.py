from yacs.config import CfgNode

_OMNIGLOT = CfgNode()
_OMNIGLOT.TEST_LANGS_LIST = ['Mkhedruli_(Georgian)', 'N_Ko',
                             'Ojibwe_(Canadian_Aboriginal_Syllabics)',
                             'Sanskrit', 'Syriac_(Estrangelo)',
                             'Tagalog', 'Tifinagh']
_OMNIGLOT.POS_NEG_RATIO = 0.5
_OMNIGLOT.DOWNLOAD = False

DATASET_DEFAULTS = {'omniglot': _OMNIGLOT}
