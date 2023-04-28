---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "No-reference quality assessment of tone mapped High Dynamic Range (HDR) images using transfer learning"
authors: ["V. Abhinau Kumar", "Shashank Gupta", "Sai Sheetal Chandra", "Shanmuganathan Raman", "Sumohana S. Channappayya"]
date: 2017-07-03T00:00:00-05:00
doi: "10.1109/QoMEX.2017.7965668"

# Schedule page publish date (NOT publication's date).
publishDate: 2020-08-09T16:34:44-05:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "2017 Ninth International Conference on Quality of Multimedia Experience (QoMEX)"
publication_short: "QOMEX 2017"

abstract: "We present a transfer learning framework for no-reference image quality assessment (NRIQA) of tonemapped High Dynamic Range (HDR) images. This work is motivated by the observation that quality assessment databases in general, and HDR image databases in particular are “small” relative to the typical requirements for training deep neural networks. Transfer learning based approaches have been successful in such scenarios where learning from a related but larger database is transferred to the smaller database. Specifically, we propose a framework where the successful AlexNet is used to extract image features. This is followed by the application of Principal Component Analysis (PCA) to reduce the dimensionality of the feature vector (from 4096 to 400), given the small database size. A linear regression model is then fit to Mean Opinion Scores (MOS) using L2 regularization to prevent overfitting. We demonstrate state-of-the-art performance of the proposed approach on the ESPL-LIVE database."

# Summary. An optional shortened abstract.
summary: ""

tags: []
categories: []
featured: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_pdf: "https://ieeexplore.ieee.org/document/7965668"
url_code:
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video:

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
