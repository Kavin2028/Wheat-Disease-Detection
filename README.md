# Wheat-Disease-Detection

Dataset :Kaagle-https://www.kaggle.com/datasets/sinadunk23/behzad-safari-jalal/data

So far we've trained for 3 diseases Brown_rust,Yellow_rust,Septoria
Diseases In Wheat Plant

Disease Classes:
  Aphid [Pest]
  Mite [Pest]
  Stem Fly [Pest]
  Rusts
  Black Rust / Stem Rust
  Brown Rust / Leaf Rust
  Yellow Rust / Stripe Rust
  Smut (Loose, Flag)
  Common Root Rot
  Helminthosporium Leaf Blight / Leaf Blight
  Wheat Blast
  Fusarium head blight / Scab
  Septoria Leaf Blotch
  Spot Blotch
  Tan Spot
  Powdery Mildew
  
Datset For the above diseases is-https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases

Model Used is CNN Sequential Model
Image Size-256
Batch Size-32
No of Epoch-10
Accuracy:93%
Commands:
$pip install flask
$python app.py

Screenshots:
![image](https://github.com/Kavin2028/Wheat-Disease-Detection/assets/85724232/80fba126-86dd-4af6-8b7d-4ab1073138a2)
![image](https://github.com/Kavin2028/Wheat-Disease-Detection/assets/85724232/b1151729-b6b9-411a-ac79-087fcd991dd3)

Predicted Output:
![image](https://github.com/Kavin2028/Wheat-Disease-Detection/assets/85724232/4381a73c-d21b-42f8-8d57-d598afeb504c)

Base Papers:
https://www.sciencedirect.com/science/article/pii/S2352914821001313

Run the model so the pre-trained model get downloaded n .h5 format use the pre-trained model to detect the diseases.
