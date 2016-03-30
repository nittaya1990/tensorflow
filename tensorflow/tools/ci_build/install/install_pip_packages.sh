#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

pip install sklearn
pip3 install scikit-learn

# Benchmark tests require the following:
pip install psutil
pip3 install psutil
pip install py-cpuinfo
pip3 install py-cpuinfo

# pylint tests require the following:
curl -O https://pypi.python.org/packages/3.5/p/pylint/pylint-1.5.5-py2.py3-none-any.whl
pip install pylint-1.5.5-py2.py3-none-any.whl
rm -f pylint-1.5.5-py2.py3-none-any.whl
