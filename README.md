# Abstract

This project leverages advanced deep learning techniques, including Convolutional Neural Networks (CNN) and Transfer Learning (TL), to automate the classification of brain tumors from MRI images. Utilizing the "Brain Tumor Classification (MRI)" dataset from Kaggle, we integrate the Inception V3 architecture pre-trained on ImageNet with the ESRGAN model from BasicSR for super-resolution image enhancement. This combination aims to increase diagnostic accuracy while reducing computational time and addressing critical challenges in medical imaging analysis. Our approach enhances image quality and supports rapid, reliable tumor classification, potentially improving diagnostic outcomes and treatment planning in clinical settings.

# Introduction

#### Background

A Brain tumor is considered as one of the most aggressive diseases among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. With over 11,700 annual diagnoses in the U.S. alone and a 5-year survival rate of around 34 percent for men and 36 percent for women, timely and accurate diagnosis is crucial. The primary diagnostic tool for brain tumors is Magnetic Resonance Imaging (MRI), which generates detailed images that radiologists traditionally analyze. However, this manual process is susceptible to errors due to the complex nature of brain tumors.

#### Motivation

Given the limitations of manual image analysis, there is a significant need for automated systems that enhance both the accuracy and efficiency of diagnostic processes. Recent advances in Machine Learning (ML) and Artificial Intelligence (AI) have shown promise in improving the precision of medical classifications over traditional methods.

#### Project Overview

This project introduces a system that combines super-resolution image enhancement with deep learning for the automated classification of brain tumors. We employ the Inception V3 architecture, known for its high accuracy and low computation requirements, which are crucial for medical applications where accuracy and speed are paramount. The model uses pre-trained weights from ImageNet, enabling us to leverage Transfer Learning to bypass the extensive time required for training from scratch. Additionally, we enhance the resolution of MRI images using the ESRGAN model, which quadruples the size of images while improving detail and clarity, facilitating better analysis and classification accuracy.

# Literature Review

#### Existing Research

Research in the field of brain tumor classification using MRI scans has been extensive, with over 400 projects utilizing the "Brain Tumor Classification (MRI)" dataset from Kaggle. These studies have employed a variety of deep learning architectures, achieving accuracy rates ranging from 30% to 70%. The common thread in these approaches is the reliance on raw or minimally pre-processed MRI data for classification tasks.

In parallel, the development of super-resolution (SR) and image enhancement technologies has significantly advanced, primarily driven by applications in image and video restoration, noise reduction, deblurring, and quality enhancement. Notable among these technologies are Real-ESRGAN and GFPGAN, which are practical algorithms designed for general and face image restoration, respectively. Super-Resolution Generative Adversarial Networks (SRGAN) and Enhanced SRGAN (ESRGAN) represent seminal work in this domain, with ESRGAN notably improving upon SRGAN by generating more realistic textures in enhanced images. These technologies have found applications across various fields, including enhancing security camera footage and medical imaging, significantly improving low-quality images' usability.

#### Gaps in Research

Despite the advances in both brain tumor classification and image enhancement technologies, a review of existing projects on the Kaggle dataset reveals a notable gap: they have yet to apply sophisticated image enhancement techniques directly to MRI scans before classification. This omission is critical, as enhanced imaging could reveal more detailed information crucial for accurate tumor classification. Additionally, the relatively low accuracy rates reported in existing research (30% to 70%) indicate that there is considerable room for improvement in classification models.

Our project addresses these gaps by integrating ESRGAN, a cutting-edge image enhancement model, with a high-performing CNN architecture, Inception V3. This approach aims to provide clearer, more detailed MRI images for analysis and improve overall accuracy in classifying brain tumors. By doing so, we contribute a novel methodology to the field, potentially setting a new standard for how MRI scans are processed and analyzed in medical diagnostics.

# Materials and Methods

#### Dataset

The dataset used in this project is titled "Brain Tumor Classification (MRI)" and is sourced from Kaggle. This dataset is comprised of MRI images that are instrumental in diagnosing one of the most aggressive diseases affecting both children and adults, brain tumors. These tumors account for 85 to 90 percent of all primary Central Nervous System (CNS) tumors, with approximately 11,700 new cases diagnosed annually in the United States. The dataset reflects the complexity and variability of brain tumors, including differences in size, type, and location, which are crucial for understanding tumor characteristics and for training our models.

The dataset includes a large number of MRI images, each labeled according to the type of tumor present. The tumors are categorized into several types, including benign tumors, malignant tumors, and pituitary tumors, among others. Each image in the dataset is a high-resolution MRI scan, providing detailed views of the brain structures affected by these tumors. The images vary significantly regarding the appearance and positioning of tumors, presenting a challenging yet vital resource for developing accurate classification models.

MRI is recognized as the most effective technique for detecting brain tumors due to its non-invasive nature and its ability to produce detailed images of brain anatomy. However, the manual examination of these images is often error-prone due to the complexity involved in interpreting them—a challenge compounded in regions with a shortage of skilled neurosurgeons. This dataset serves as a critical tool for training our CNN and utilizing Transfer Learning (TL) for enhanced tumor classification. It also supports our goal of automating the detection and segmentation process of brain tumors. By doing so, we aim to improve diagnostic accuracy and reduce the time needed for MRI analysis, particularly in resource-limited settings.

Application: The project utilizes Convolutional Neural Networks (CNNs) and Transfer Learning (TL) as part of a deep learning approach to detect, classify, and segment brain tumors in MRI images. This automated system is designed to enhance and support diagnostic processes, potentially alleviating the burden on medical professionals in both developed and developing countries.

#### Preprocessing

The dataset, initially divided into 'Training' and 'Testing' folders, includes subdirectories named 'glioma_tumor', 'no_tumor', 'meningioma_tumor', and 'pituitary_tumor'. Each directory is labeled according to the type of brain tumor (or absence thereof) the images within represent. To ensure comprehensive learning and validation, we combined both the training and testing datasets into a single dataset.

We implemented an image deduplication step to improve the model's performance and ensure it learns from unique samples. Each image was converted into a hash value, and we used a hashing algorithm to store these values. We then employed a conditional check to verify if an image's hash was already in our dataset hash store, thus eliminating duplicates. This process ensures that our training and testing data are comprised solely of unique images, preventing the model from being biased toward frequently repeated data.

All images were loaded into a numpy array for processing, ensuring they were appropriately formatted for input into our deep learning models. Corresponding labels for each image, derived from their directory names, were also stored in a separate numpy array. This approach maintains an organized and accessible dataset structure, which is crucial for effective model training.

Finally, to minimize any potential bias resulting from the order of data, we randomized the entire dataset. This randomization ensures that the distribution of data into training, validation, and testing sets does not skew model performance metrics, providing a more accurate assessment of the model's true classification ability.

## Image Enhancement Techniques

### Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN)

For this project, we employed ESRGAN, a cutting-edge deep-learning model developed to improve image resolution significantly. Traditional super-resolution techniques often produce blurry and lack detail, but ESRGAN uses a Generative Adversarial Network (GAN) framework to enhance image quality by generating realistic, high-resolution images from low-resolution inputs.

#### Architecture

ESRGAN comprises two main components: the generator and the discriminator. The generator is tasked with creating high-resolution images that are indistinguishable from true high-resolution samples. At the same time, the discriminator evaluates the authenticity of these generated images, distinguishing between actual and generated outputs. This adversarial process drives the generator towards producing more accurate and high-quality images over time. This method produces images that are not only high-resolution but also visually appealing and realistic. ESRGAN has found applications in various fields, including upgrading low-quality images from security cameras and enhancing the resolution of medical imagery.

Training ESRGAN involves an iterative adversarial process: The generator creates images that fool the discriminator into believing they are genuine high-resolution images. The discriminator learns to differentiate between the real high-resolution images and those produced by the generator, providing essential feedback to enhance the generator's output.

#### Advantages of ESRGAN

- Produces textures and details that are significantly more realistic and refined compared to previous super-resolution methods.
- Capable of training with unpaired data, which is particularly advantageous when exact high-resolution counterparts are not available.
- Flexible across various types of imagery, including detailed medical imaging needed for accurate diagnostic processes.

#### Implementation for MRI Images

For our project, we used a pre-trained ESRGAN model from BasicSR to enhance the quality of MRI brain images. We developed a specific preprocessing routine to adapt the MRI images for compatibility with ESRGAN. Images were normalized to a [0, 1] scale to maintain consistency with the training conditions of the pre-trained model. MRI images, typically in grayscale, were converted to RGB format (3 channels) to match the input requirements of ESRGAN. We reshaped Images to include a batch dimension, facilitating processing through the neural network. Finally, the preprocessed numpy arrays were converted into tensors, PyTorch's standard input format for deep learning models.

Through these steps, ESRGAN was effectively applied to enhance the resolution and detail of MRI images, ensuring that the subsequent classification by the CNN could be performed with greater accuracy and reliability.

### Other Image Transformations

In our quest to enhance the detail and clarity of MRI images for more accurate brain tumor classification, we explored several image processing techniques alongside ESRGAN. Each technique was tested for its potential to improve image quality and highlight critical features necessary for effective diagnosis.

#### Laplacian Filtering

Laplacian filtering, a second derivative technique, was applied to the MRI images to enhance edges and features. This method highlights regions of rapid intensity change, which are crucial in delineating the boundaries of brain tumors. However, we observed that applying the Laplacian filter led to an undesirable increase in image noise, detracting from the overall image quality necessary for precise classification.

#### Histogram Equalization

We tested histogram equalization to improve the contrast of the MRI scans. This technique redistributes the pixel intensity values to stretch out the intensity range of the image. Although applicable in some other contexts, standard histogram equalization tended to over-enhance contrast in some areas while under-enhancing in others, leading to non-uniformity in image quality.

#### Contrast Limited Adaptive Histogram Equalization (CLAHE)

Recognizing the limitations of standard histogram equalization, we employed CLAHE, which improves contrast by adaptively adjusting the histogram equalization over small regions or tiles within the image. This method provided a more balanced enhancement, improving local contrast without the noise amplification typically seen with global histogram equalization techniques.

#### Gaussian Blur

Gaussian blur was applied as a smoothing filter, reducing noise and detail by averaging pixel values with a Gaussian function. While effective for removing noise, this technique also blurred essential details and edges, which are vital for accurately identifying and classifying brain tumors.

#### Data Augmentation Techniques

Data augmentation methods like flipping images along the y-axis and applying slight rotations were considered to increase the diversity of training data. However, considering the nature of brain MRI scans, often taken from similar angles and directions, these techniques did not significantly contribute to model performance and would have extended training times unnecessarily.

#### Final Decision

After evaluating these techniques, we decided to integrate only CLAHE and the pre-trained ESRGAN model from BasicSR into our pipeline. CLAHE provided the necessary contrast enhancement without compromising image quality, while ESRGAN significantly improved the resolution and clarity of the images. The combination of these two methods proved most effective in enhancing the MRI images, facilitating more accurate and reliable tumor classification by our CNN model. While beneficial in specific scenarios, other techniques did not meet the project's requirements for maintaining high-quality, detailed imaging necessary for precise medical diagnosis.

## Convolutional Neural Network Architecture

#### Overview of Inception v3

Inception v3 is a sophisticated deep convolutional neural network developed primarily for image classification tasks. Known for its efficiency and accuracy, this architecture builds on the success of previous models by incorporating advanced features that allow it to perform exceptionally well across various image recognition tasks.

#### Architecture Details

At the core of Inception v3 are the Inception modules (types A, B, C, D, E, and an auxiliary module). Each module consists of several parallel paths of convolutions, pooling operations, and concatenations. This structure enables the model to capture and integrate features at various scales and complexities, making the network robust in recognizing diverse patterns in the data.

BasicConv2d is a fundamental component used across different Inception modules. It encapsulates a convolutional layer followed by batch normalization and ReLU activation function. This combination is repeated throughout the network to maintain non-linearity and normalize the activations, helping to speed up the training process without significant overfitting.

#### Network Progression

The model starts with standard convolutional and pooling layers that initially reduce the input image's spatial dimension, followed by the sequential application of Inception blocks (from A to E), which allows the model to develop a deeper and more nuanced understanding of the image features. Following the final Inception blocks, the network employs average pooling to reduce feature dimensions. Then, it flattens the output for the following stages, and a dropout layer is used to prevent overfitting, followed by a fully connected layer that outputs the final classification predictions.

#### Auxiliary Classifier

One unique aspect of the Inception v3 architecture is its use of an auxiliary classifier. This secondary output helps stabilize the training in the deeper network by providing additional gradient signals in intermediate layers:

- **Main Output:** The primary output from the network that provides the final decision signals used for classification tasks.
- **Auxiliary Output:** Active only during training, this output aids by acting as an intermediary regularization step. It helps the model maintain healthy gradients throughout training, mainly benefiting the learning of earlier layers.

The loss from the auxiliary classifier is scaled down (typically by a factor such as 0.4) when combined with the main loss to ensure it does not dominate the training process but still contributes effectively.

#### Adapting to Dataset Specifics

The original Inception v3 architecture, as provided by PyTorch, is configured to classify images into one of 1000 different categories, matching the requirements of the ImageNet competition for which it was designed. However, our project's dataset contains only four distinct classes of brain tumors. To accommodate this, we significantly modified the network's final classification layer.

- **Original Setup:** The typical final layer in Inception v3 is a fully connected (linear) layer designed to output 1000 class predictions corresponding to the 1000 categories of ImageNet.
- **Modified Setup:** To tailor the network for our specific task, we replaced the original fully connected layer with a new sequence of layers designed to optimize performance for a four-class problem.

**Dropout:** The first modification is the addition of a dropout layer. Dropout is a regularization technique used to prevent overfitting in neural networks by randomly setting a fraction of input units to zero during training. This helps to make the model robust to unseen data.

**Linear:** After dropout, we introduced a new fully connected (linear) layer. This layer maps the learned features to a new space that better represents the distinctions between the four tumor classes.

**ReLU:** We included a ReLU (Rectified Linear Unit) activation function following the first linear layer. ReLU helps to introduce non-linearity into the model, enabling it to learn more complex patterns.

**Dropout:** Another dropout layer follows the ReLU activation to enhance the model's generalization capabilities further.

**Final Linear:** The sequence concludes with a second fully connected layer that outputs the final predictions for the four classes. This layer directly maps the processed features to our four desired outputs.

These modifications were driven by the need to adapt the model for high accuracy in a much smaller classification problem than initially designed. Reducing the complexity of the output layer to match the number of target classes reduces the risk of overfitting and improves the model's ability to generalize from training data to real-world applications. Additionally, introducing additional dropout layers increases the robustness of the model, an essential factor in medical diagnostic applications where reliability is paramount.

#### Implementation Challenges

During our project, the dual output nature of Inception v3 main and auxiliary presented initial challenges, particularly since the pre-trained PyTorch model required this feature. Understanding and implementing the correct handling of both outputs were crucial for effectively leveraging the architecture’s full capabilities, especially in training dynamics where both outputs influence the loss calculations and model updates.

## Training Process

#### Phased Training Approach

We employed a strategic phased training approach to effectively leverage the pre-trained Inception v3 model for our specific classification task. Initially, we froze all layers except for the final fully connected ('fc') layer, which directly affects output predictions. This layer was first unfrozen to adapt the model's outputs to our specific classes immediately. We then progressively unfroze deeper layers, starting from 'Mixed_7c' in Phase 2, moving to 'Mixed_7b' in Phase 3, and 'Mixed_7a' in Phase 4. This gradual unfreezing helps fine-tune the more abstract features relevant to our dataset while maintaining the integrity of the universally applicable features learned earlier in the network.

The underlying philosophy for this approach is based on the neural network learning hierarchy. Lower layers, which learn universal features such as edges and textures, were kept frozen to retain their pre-trained capabilities. Higher layers, closer to the output, were gradually trained to adapt to more specific features pertinent to brain tumor MRI images. By fine-tuning only the top layers, we reduced computational requirements and expedited the training process, a crucial factor given the hardware constraints.

#### Optimization and Training Dynamics

We defined a dedicated training function to handle the complexities of phased training. This function managed the unfreezing of layers as per the defined phases. It also monitored and recorded accuracy for both training and testing datasets and tracked the training loss. This function dynamically adjusted the learning rate of the optimizer to refine training as progress was made, optimizing the convergence rate. At the start of each phase, the optimizer was reinitialized to update only the unfrozen layers, ensuring focused and effective training.

We used Adam Optimizer to update the model weights. Adam is favored for its adaptive learning rate capabilities, which help converge faster and more effectively in training deep neural networks. This choice is particularly beneficial for fine-tuning pre-trained models as it gently adjusts the weights in gradually unfrozen layers, minimizing the risk of distorting already learned features.

We selected the Cross Entropy Loss as the criterion for the training process. This loss function is well-suited for classification problems with multiple classes, as it measures the performance of a classification model whose output is a probability value between 0 and 1. Cross Entropy Loss increases as the predicted probability diverges from the actual label, making it an effective measure for our task of classifying brain tumors into four categories.

#### System Specifications

We trained the model on a laptop manufactured as a “gaming laptop,” which provided a robust platform for handling the intense computational demands of training a deep neural network.

- GPU: NVIDIA GeForce RTX 2060
- CPU: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- RAM: 16.0 GB
- System Type: 64-bit operating system, x64-based processor

#### Training Execution

We trained the model on GPU over 100 epochs, distributed across 4 phases (25 epochs per phase), totaling approximately 4 hours of training time. This phased approach allowed for meticulous refinement of the model across different levels of abstraction.

We trained two versions of the model, one using the enhanced images processed by ESRGAN and CLAHE and another using the original, unprocessed images. This comparative approach allowed us to directly assess the impact of our image enhancement techniques on the model's performance.

## Methodology for Evaluating the Model

#### Model Outputs and Loss Calculation

The Inception v3 architecture generates two types of outputs during training: the main classifier output and an auxiliary output. The auxiliary output serves primarily during the training phase to stabilize the training process through additional regularization and error correction.

- **Main Output:** This is the primary channel through which the model's performance is determined. The loss calculated from this output (main_loss) is pivotal for the model’s learning and adjustment.
- **Auxiliary Output:** Active during training, the auxiliary loss (aux_loss) is scaled down (typically by a factor of 0.4) to ensure it supports but does not overshadow the main loss. It aids in the model's convergence without dominating the training dynamics.

The total training loss is a composite of these two losses, where the auxiliary loss contributes less significantly due to its lower weighting. This structured approach allows the model to benefit from the auxiliary features without compromising the primary learning objectives.

#### Accuracy Calculation and Metrics

During testing and real-world deployments, the auxiliary classifier is disregarded, and the model's predictions are based solely on the main output. Accuracy is computed by comparing the predicted class (the class with the highest probability from the main output) against the actual class labels. This is typically done using ‘torch.max()’ on the logits to identify the predicted classes, which are then matched against the true labels to determine the model’s accuracy.

#### Performance Evaluation Functions

To systematically assess the model's performance throughout training and upon completion, we developed several vital functions:

- **Training Function:** This function outputs the training loss and accuracy metrics for both the training and testing datasets as the model progresses through epochs. Each epoch includes an evaluation phase to calculate and record these metrics.
- **Evaluation Function:** This function specifically measures and returns the accuracy of the model on any given dataset (training or testing), facilitating ongoing monitoring of the model’s performance.
- **Visualization Tools:** This function plots the training loss and both training and testing accuracies against the total number of epochs. This visual representation helps understand the model's learning curve and pinpoints any epochs where significant improvements or regressions occur.

#### Classification Reporting and Confusion Matrix

To provide a detailed analysis of the model's classification capabilities:

- **Report Function:** Utilizes scikit-learn's utilities to generate a detailed classification report, which includes precision, recall, f1-score, and support for each class. This function provides a deeper insight into the model’s performance across different metrics.
- **Confusion Matrix:** Also generated through scikit-learn, the confusion matrix visualizes the accuracy of the model’s predictions. It shows how well the model distinguishes between classes, highlighting any consistent misclassifications (confusion) between different tumor types.

These tools collectively offer a comprehensive framework for evaluating the model’s effectiveness in classifying brain tumors, ensuring any deficiencies are identified and addressed before deployment.

# Results

### Model Training and Evaluation with Enhanced Images

<div style="text-align: center;">
<img src="expalin_Images\EI_plot.jpg" alt="EI Plot">
</div>

Throughout the training of our enhanced Inception v3 model, we observed significant improvements in both training and testing accuracy, as shown in the detailed epoch-by-epoch plot:

- **Phase 1:** Training began with a loss of 1.6 and an initial training accuracy of 83.43%. Over 25 epochs, both the training loss decreased, and the train and test accuracies improved, stabilizing around 86.15% and 81.53%, respectively.
- **Subsequent Phases:** With further training and the progressive unfreezing of additional layers, the model showed remarkable improvement:
- By the end of Phase 2, the training loss was reduced to 0.5868, and both train and test accuracies reached near-perfect and 89.90%, respectively. This trend continued into Phases 3 and 4, with test accuracy peaking at 90.59%.

<div style="text-align: center;">
<img src="expalin_Images\EI_table.jpg" alt="EI Table">
</div>

<div style="text-align: center;">
<img src="expalin_Images\EI_matrix.jpg" alt="EI Matrix">
</div>

The Classification Report indicates that our model demonstrates high precision when predicting Classes 0 and 3, suggesting that predictions made by our model for these classes are usually accurate.

- This observation is supported by the Confusion Matrix, which shows that the model misclassifies true instances of Class 0 as Class 1, and similarly, misclassifications involving Class 3 are infrequent.
- Although the precision for Classes 1 and 2 is lower compared to Classes 0 and 3, the results for these classes are still reasonably good.

Class 0 shows a low recall compared to the other classes, indicating that the model frequently overlooks actual instances of Class 0. Consequently, many instances of Class 0 are likely misidentified as belonging to other classes, resulting in many false negatives for Class 0.

### Model Training and Evaluation with Original Images

<div style="text-align: center;">
<img src="expalin_Images\OR_plot.jpg" alt="OR Plot">
</div>

The model trained on original images showed distinct patterns in training loss and accuracy compared to the enhanced model:

- **Phase 1:** It was initiated with a training loss of 1.6 and a training accuracy of 86.45%. Test accuracy started at 56.59% and gradually increased to 62.11% by the end of this phase.
- **Subsequent Phases:** Continued training stabilized training accuracy at 100%, indicating the model effectively learned from the training data. However, test accuracy hovered around 71% to 75%, peaking at 75.06% in Phase 3.

These results suggest that while the model learned to predict the training data accurately, it struggled to generalize effectively to unseen data, reflected in the relatively lower test accuracies.

<div style="text-align: center;">
<img src="expalin_Images\OR_table.jpg" alt="OR Table">
</div>

<div style="text-align: center;">
<img src="expalin_Images\OR_matrix.jpg" alt="OR Matrix">
</div>

The Classification Report indicates that our model demonstrates perfect precision when predicting Classes 0 and 3, suggesting that our model's predictions for these classes are very accurate.

- This observation is supported by the Confusion Matrix, which shows that the model does not misclassify instances from these two labels.
- Although the precision for Classes 1 and 2 is very low compared to Classes 0 and 3, the results for these classes are still reasonably good.
  - Class 1 is classified as Class 0 by the model.
  - Class 2 is usually classified as class 0 by the model.

Class 0 shows a very low recall compared to the other classes, indicating that the model frequently overlooks actual instances of Class 0. Consequently, many instances of Class 0 are likely misidentified as belonging to other classes, resulting in many false negatives for Class 0.

## Summary

Our project introduces an innovative methodology to the field by integrating ESRGAN, a state-of-the-art image enhancement model, with the highly capable CNN architecture, Inception V3. This fusion is designed to produce clearer, more detailed MRI images for analysis, aiming to elevate the overall accuracy in the classification of brain tumors. Such an approach has the potential to redefine standards for processing and analyzing MRI scans in medical diagnostics, specifically by employing super-resolution as an image enhancer in the data preparation stage for image classifiers.

Building on this, the model trained on original images showed a respectable performance with an accuracy of 74%, which is within the upper range of typical results for this dataset on Kaggle, where accuracies generally ranged from 30% to 70%. However, the analysis underscores the pivotal effect of image enhancement on model performance, as evidenced by a substantial increase in accuracy to 90% with the use of enhanced images. This notably improves the model's ability to generalize to unseen data. While the model demonstrates high precision yet low recall for certain classes, indicating its reliability in identifying specific conditions and its tendency to miss other true cases, this revelation emphasizes the necessity for additional refinement. It suggests a continued investigation into image preprocessing techniques may be beneficial in order to optimize diagnostic accuracy and mitigate the current limitations in detection.
