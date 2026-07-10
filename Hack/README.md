# Hack Resources

[Slide deck for Friday presentations -- will be inserted later!](TODO)

[google drive for data](https://drive.google.com/drive/u/1/folders/13ySEme-B8XDMYgTZ8_rVpMarRUUGYbTw)

It is most straightforward to directly work with the above Google drive from colab,
since the files don't need to leave Google's servers in that case.
The way I figured out how to do this is as follows (there might be a better method):
1. open the above Google drive.
2. right-click on the file or folder you need.
3. click "Organize" -> "Add shortcut".
4. in "All locations", choose "My Drive" and click "Add".
5. in the colab instance, open the "Files" explorer on the left.
6. click "Mount Drive" icon.
7. you should be able to see the file now.
   To switch the colab working directory to your drive, type:
   ``cd "/content/drive/MyDrive/"``

## (1) Image Classification / Regression / Generation Tutorials and Exercises
Analyzing Cherenkov light from massive neutrino detectors, such as Super-Kamiokande,
with convolutional neural nets.

All the material is available at:
[this link](https://github.com/cesarjesusvalls/A3NetRingImg)

contact person: Vera Maiboroda

## (2) Galaxy Cluster Masses with CNNs
please follow [this link](https://drive.google.com/drive/folders/1yLN3xN-1EU2HPYcbX68Z8SgLQcs1LpiI?usp=sharing)

Goal: the paper, notebook, and data set in the link above are for a project that uses a CNN to estimate the mass of a galaxy cluster from x-ray mock observations. In this hack, you will first recreate the paper’s results. Then you will apply a rotationally invariant CNN, and see if that approach improves the results!

contact person: Leander Thiele

## (3) Galaxy redshifts with Decision Trees
please follow [this link](https://archive.stsci.edu/hello-universe/3d-hst)

Goal: the notebook and data set in the link above are for a project that uses a decision tree to estimate the redshifts of galaxies. In this hack, you will explore other ML algorithms for using the same data to estimate galaxy redshifts. Scikit-Learn has a [flow chart](https://scikit-learn.org/1.3/tutorial/machine_learning_map/index.html) for selecting ML methods. Support Vector Regression might be an interesting technique to try.
Note: the Hello Universe notebook is also available on the [TIKE science platform](https://timeseries.science.stsci.edu/hub/spawn) (select the default environment to spin up an AWS server)

contact person: Vera Maiboroda


## (4) Interpretation of GAN's latent space

In the Thursday lecture, generative models such as VAE, GAN, flow-based models, and diffusion models will be introduced. Among these, VAE and GAN allow for the dimensionality of the latent space to be set smaller than that of the data space. VAE is sometimes used for dimensionality reduction, and researchers often attempt to interpret its latent space after training (see e.g., [this work](https://ui.adsabs.harvard.edu/abs/2020AJ....160...45P/abstract)). The goal of this project is to gain a deeper understanding of the latent space of GANs.

- Key Questions: Can the latent space of a GAN be interepreted? Can we design a GAN with interepretable latent space?

Unlike VAE, GANs do not provide a direct mapping from the data space to the latent space. A brute-force approach to interpret the latent space of a GAN might involve generating a dataset, estimating the parameters that could correspond to the generated data, and then identifying which dimensions of the latent space correlate with those parameters.

It is also possible to design the model so that the latent space is interpretable by making the model conditional, i.e., by using the parameter space as the latent space.

Materials: Example codes for VAE and GAN can be found in the notebooks:

[VAE](https://colab.research.google.com/drive/1oscMuJK7QlifUNMGnayZQp2js9cx7OYi?usp=sharing)

[GAN](https://colab.research.google.com/drive/10Zb_yX1iGXYXwt6BGqCLz3rIkGjT1_B1?usp=sharing)

[Flow](https://colab.research.google.com/drive/1AA7fD5K-zYjkLqZERBDWinbeKO2BHTEE?usp=sharing)

Data are: SDSS spectrum dataset in [Google drive](https://drive.google.com/drive/u/1/folders/13ySEme-B8XDMYgTZ8_rVpMarRUUGYbTw).


Note: I recommend using low-dimensional data (e.g., 1D) and reducing the size of data (e.g., by lowering the resolution or clipping the data) whenever possible, given the limited computational cost and time.

Contact person: Leander Thiele

## (5) Transfer learning 

Transfer learning is a machine learning technique where a model trained on one task is adapted and reused for a different but related task, leveraging the knowledge gained from the initial task to improve performance on the new one. 

- Question: Is transfer learning always effective? 

In PyTorch, pre-trained weights are available for various image-related tasks, such asclassification and semantic segmentation ([link](https://pytorch.org/vision/stable/models.html)). These models are pre-trained on "normal" images. It would be interesting to investigate whether further training these models with astronomical data (e.g., 2D images of galaxies and large-scale structures) offers any advantages compared to training from scratch (and when it is so). If that is the case, it would mean that we should always utilze some kind of pre-trained model whenever we do research using ML! Exploring whether pre-trained models can be directly applied to astronomical data without transfer learning might be also interesting. Example codes of transfer learning can be found [here](
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

I have not tested if the pre-trained PyTorch models for images run smoothly in Google Colab. If computational resources are not enough, you can switch to 1D data. I am not certain if there are pre-trained models for 1D dataset available online, but if not, you can create a custom approach by splitting the data you have and pre-trainining the model on one subset, then applyting transfer learning to the other (e.g., using z < 1 galaxies for pre-training,and z > 1 galaxies for transfer learning).

Note: if a large amount of data is used in transfer learning, the differences between transfer learning and training from scratch may become negligible. To properly assess the benefits of transfer learning, it may be helpful to limit the size of the dataset used for transfer learning. See, e.g., Fig. 5 of [this study](https://arxiv.org/abs/2310.02994) for similar investigation.

Contact person: Adrian Bayer

## (6) Weather classifier

Develop a CNN classifier of the sky condition using images obtained with our sky monitor camera.

[Background and goal](https://drive.google.com/file/d/1_IEtQnvjdHcb9rvKcwbl7F1B4z4yU4i8/view?usp=sharing)

[Data](https://drive.google.com/drive/folders/1KJs-OVU-ZSyiTIHtZusgFAktwFtv97S9?usp=share_link)

There are thousands of images in the above link (google drive)~~, while I have ten thousands of images in my USB flash drive. If you have try to use them, please let me know~~. 

Contact person: Sungwook Hong

## (7) Variable selection

Find a set of variable to construct the best model of the magnitude at maximum of Type Ia supernovae using LASSO + Cross-validation. This theme is not for deep learning and neural network related methods, while you may try them for this theme.

[Background and goal](https://drive.google.com/file/d/1b0wLOvDtJHA8rZJKHdp-XUyXspX65q2g/view?usp=share_link)

[Data](https://drive.google.com/drive/folders/1kz94LOxBTEpKeNBrG56K8rs9sp2u9Y8M?usp=sharing)

Contact person: Adrian Bayer

## (8) Using MLs to infer the fundamental parameters of the Universe

For this hack, we're using the CAMELS multifield data set.
The data is in the `CAMELS_multifield` directory in the Google Drive linked above.

Please see the [notebook](CAMELS_multifield.ipynb) for a quick explanation of the data format.

Use the available data to train a machine learning to infer the underlying cosmological parameters
from the images.

How well does your trained network generalize? For example, if you train on the IllustrisTNG
simulations and test on SIMBA, are the inferred cosmological parameters correct?

Or, can you determine which simulation was used to make a given image?

contact person: Leander Thiele, Sungwook E. Hong

## (9) A generative model for galaxy spectra

When testing data processing and analysis pipelines, it is often very useful to have realistic simulations
of the anticipated data.
One way of producing such simulations is by learning the probability distribution of existing data and
drawing new samples from it.

Such learning can be performed with any generative model (normalizing flow, GAN, VAE, ...).
In this hack, you're free to choose which architecture you want to use.

The goal is to generate synthetic galaxy spectra, by learning from measured Sloan Digital Sky Survey data.
The data can be found in the google drive linked above.

You can choose to leave the model completely free to generate spectra, or condition
it on parameters such as redshift and metallicity.
Such conditioning can be useful in practice in order to perform domain shifts.

Please see the [notebook](SDSS_spectra.ipynb) for a quick explanation of the data format.

*Ideas*: going full deep learning is not always the best idea. Sometimes we can capture a lot of the data with
something simpler, like principal component analysis, already.
If you're having trouble getting your model to work, this may be an option.
Also consider downsampling the spectra for a first prototype.
You could use then try to fill in the fine details with a second stage.
*This is really an open-ended hack and you can be creative!*

contact person: Leander Thiele


## (10) A generative model for galaxy image

During the lectures, we use the [galaxy image dataset with morphology classification](https://drive.google.com/uc?id=1QjTECPvkzrXUJxishXS-0KTknNq0erly).

While we have been using this dataset mostly for galaxy morphology classification, can we use it in a reversed way?
That is, can we generate a mock galaxy image from a given morphology?

Moreover, what other parameters can we introduce for our mock galaxy image generation, such as inclination angle, major/minor radius, and RGB channel brightness?

contact person: Sungwook E. Hong

## (11) Automatic detection of defects on detector hardware

In preparation for the experiments, we often need to produce specific detector hardware parts. When the production scale becomes large, there is a need for a tool capable of automatically performing quality control on hardware components. The goal of this hack is to detect defects in the metallic contact parts.

The data can be found [here](https://cernbox.cern.ch/s/RrahTLjGbYufqnB).

An example of where this can be used is the High-Luminosity upgrade of the LHC, which involves replacing some of the subdetectors. For instance, the Inner Detector in the ATLAS experiment will be replaced by an All-Silicon Tracker (ITk). One of the main components of the ITk is the pixel module, which records the passage of particles through it. For the ITk, over 10,000 pixel modules need to be assembled. With such large quantities, human visual inspection of every single module is not feasible, so we are looking for ways to automate this procedure.

contact person: Vera Maiboroda

## (12) CosmoBench -- Benchmarking ML for Cosmology

How well do different ML methods perform at cosmological inference? Are linear models enough? Do we gain a lot by using neural networks? We recently published a [paper](https://arxiv.org/abs/2507.03707) benchmarking (comparing) different methods for cosmological inference, including linear models for the power spectrum and Graph Neural Networks, but there are so many methods and tasks to be explored!

The data with tutorials is described [here](https://cosmobench.streamlit.app/). It consists of simulated of galaxy, halos, and more! This is a very open project where you can try the methods in the paper, or new methods ad architectures, to produce better predicitons of cosmological parameters from the data, and also the scaling with number of simulations. 

contact person: Adrian Bayer
