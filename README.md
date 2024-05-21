<p align='center'>
  <a href='https://praig.ua.es/'><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
  <a href='https://www.litislab.fr/'><img src='graphics/Litis_Logo.png' alt='LITIS-logo' width='100'></a>
</p>

<h1 align='center'>Sheet Music Transformer++: End-to-End Full-Page Optical Music Recognition for Pianoform Sheet Music</h1>

<h4 align='center'><a href='https://arxiv.org/abs/2405.12105' target='_blank'>Full-text preprint</a>.</h4>

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.12.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white' alt='Lightning'>
  <img src='https://img.shields.io/static/v1?label=License&message=MIT&color=blue' alt='License'>
</p>

<p align='center'>
  <a href='#about'>About</a> •
  <a href='#how-to-use'>How To Use</a> •
  <a href='#citations'>Citations</a> •
  <a href='#license'>License</a>
</p>

## About

This GitHub repository contains the implementation of the Sheet Music Transformer ++ (SMT), the upgraded version of the Sheet Music Transformer model for full-page pianoform music sheet transcription. Unlike traditional approaches that primarily resort this challenge by implementing layout analysis techniques with end-to-end transcription, the SMT ++ model offers a image-to-sequence solution for transcribing these scores directly from images. To do so, this model is trained through a progressive curriculum learning strategy with synthetic generation.

<p align="center">
  <img src="graphics/smt++.jpeg" alt="content" style="border: 1px solid black; width: 800px;">
</p>

:warning: **Please bear in mind that, although some results have been published, this is still an work-in-progress project, bugs may be found**

## How to use

Usage instructions and data publication coming (hopefully) soon!

## Citations

```bibtex
@misc{RiosVila:2024:SMTplusplus,
      title={Sheet Music Transformer ++: End-to-End Full-Page Optical Music Recognition for Pianoform Sheet Music}, 
      author={Antonio Ríos-Vila and Jorge Calvo-Zaragoza and David Rizo and Thierry Paquet},
      year={2024},
      eprint={2405.12105},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
