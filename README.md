# DIRECT: Deep Reinforcement Learning for Tourist Route Generation

This repository contains the source code for **DIRECT: Deep Reinforcement Learning for Tourist Route Generation** and scripts to fetch data from [OpenStreetMap](https://www.openstreetmap.org/).


## Dependencies
The `requirements.txt` has been provided for installing all dependencies in a virtual environment. To install the requirements run:
```
pip install -r requirements.txt
```

Dependencies include:
- Python 3.12.4
- PyTorch 2.7.0
- `stable-baselines3`
- `gymnasium`


## DIRECT Model

- Run script `train_eval_model.py` to train the model on training set and compute evaluation results on the test set. Specify the city, candidate POI generator k and alpha parameters in the arguments.
```
python3 train_eval_model.py --city '<city_name>' --cpg_k '<cpg_k>' --alpha_params '0.33,0.33,0.33'
``` 
- DIRECT model checkpoints and logs will be saved in the `content` folder. The test set evaluation metrics and generated routes will be saved in `results` folder.

## Disclaimer

- The OpenStreetMap name is a trademark of the OpenStreetMap Foundation and is used with their permission. We are not endorsed by or affiliated with the OpenStreetMap Foundation.
- We do not include data from OpenStreetMap in this repository. OpenStreetMap data is available under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) https://www.openstreetmap.org/copyright.
- Verona dataset by Dalla Vecchia et al. (MIT License) is available at https://github.com/smigliorini/itinerary-drl
- The tag list in `dataset_generation/constants.py` contains OpenStreetMap tags, which are licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) by the OpenStreetMap Foundation https://www.openstreetmap.org/copyright.


## Citation
If you find this work useful in your research, please cite:

```bibtex
@article{mann2026,
  author  = {Mann, Genivika and Dadwal, Rajjat and Demidova, Elena},
  title   = {DIRECT: Deep Reinforcement Learning for Tourist Route Generation},
  journal = {ACM Transactions on Spatial Algorithms and Systems},
  note = {Manuscript accepted for publication}
}
```


