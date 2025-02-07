# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

from .classifier_light import classifier_config_dict_light
from .classifier_mdr import tpot_mdr_classifier_config_dict
from .classifier_sparse import classifier_config_sparse
from .classifier_nn import classifier_config_nn
from .classifier import classifier_config_dict
from .clustering import clustering_config_dict
from .regressor_light import regressor_config_dict_light
from .regressor_mdr import tpot_mdr_regressor_config_dict
from .regressor_sparse import regressor_config_sparse
from .regressor import regressor_config_dict
