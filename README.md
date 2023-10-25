<p align="center">
  <img width=70% src="https://ollieboyne.github.io/FOUND/images/logos/found_v1.png">
</p>

# FOUND

> **FOUND: <ins>F</ins>oot <ins>O</ins>ptimisation with <ins>U</ins>ncertain <ins>N</ins>ormals for Surface <ins>D</ins>eformation using Synthetic Data**  \
> Winter Conference on Applications of Computer Vision 2024 \
> [Oliver Boyne](https://ollieboyne.github.io), [Gwangbin Bae](https://www.baegwangbin.com/), [James Charles](http://www.jjcvision.com), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]]() [[project page]](https://ollieboyne.github.io/FOUND/)


## Quickstart

1) `git clone --recurse-submodules http://github.com/OllieBoyne/FOUND`
2) Install dependencies: `pip install -r requirements.txt`
3) Download the [pretrained FIND model](https://drive.google.com/drive/folders/1XWmEVo3AdnhJU2fs6igls-emp93beQpm?usp=share_link) to `data/find_nfap`
4) Download our [benchmark foot dataset]()
5) Run a foot to a scan:

```
python FOUND/fit.py --exp_name <exp_name> --data_folder <data_folder>
```

You can also define `--cfg` and link to a `.yaml` file to change the default parameters. See `FOUND/utils/args.py` for all arguments.


## Data

We provide our synthetic foot dataset, [SynFoot](https://github.com/OllieBoyne/SynFoot), which contains 50K synthetic foot scans, with RGB, normals, and masks.

We also provide a benchmark evaluative dataset, [To be added]


### Related work

Please check out all of our projects that built into this work!

- [FIND - Generative foot model](https://ollieboyne.github.io/FIND)
- [Surface Normal Estimation w/ Uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty)
- [BlenderSynth - Synthetic data generation](https://ollieboyne.github.io/BlenderSynth)


### Citation

If you use our work, please cite:

```
@inproceedings{boyne2023found,
            title={FOUND: {F}oot {O}ptimisation with {U}ncertain {N}ormals for Surface {D}eformation using Synthetic Data},
            author={Boyne, Oliver, and Bae, Gwangbin, and Charles, James and Cipolla, Roberto},
            booktitle={Winter Conference on Applications of Computer Vision (WACV)},
            year={2024}
}
```