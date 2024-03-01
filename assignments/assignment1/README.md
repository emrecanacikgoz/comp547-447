# Assingment 1: Autoregressive Models

**Due March 15, 11:59pm**

In this homework, you will implement and train a variety of different autoregressive models, such as PixelCNN and iGPT. The homework consists of two components:
* `assignment1.ipynb` : Contains all of the coding and written inline questions, and will automatically generate and display results for you after completing each question. You will submit the notebook to BB after completing the homework. Open it on Colab by clicking on the file, and then "Open in Colab" at the top. **Submit a PDF version of the notebook to the code (Print -> Preview -> Save) on BB in the assignment with (code)**
* `assignment1_latex` :  Contains LaTeX files and figures needed to generate your PDF submission to Gradescope. Copy the images saved from the notebook into the `figures/` folder and fill out the empty test losses.  **Submit the Latex PDF in the assignment with (PDF)**

You can open the notebook in Google Colab to get access to a free GPU, or you can link Colab to a local runtime to run it on your own GPU. 


# Table of Contents
* *Question1: 1D Data (25 Points)*
* *Question2: PixelCNNs (45 Points)*
* *Question3: Casual Transformer - iGPT (30 Points)*
* *Bonus: Casual Transformer - Tokenized Images (25 Points)*

_For bonus question, you can find the vqvae checkpoints [here](https://drive.google.com/drive/folders/1Gfk-OOzmOXQ0J3WbQU7jy99Xk4gNfirw?usp=sharing). You must put them inside your `./data/hw1_data/` folder._


# Assignment Policy

In our Deep Unsupervised Learning course, all work on assignments must be done individually unless stated otherwise. You can make discussions in an "abstract" way. However, it's crucial to distinguish between constructive discussions and the inappropriate use of resources: turning in someone else’s or your friend's work, in whole or in part, as your own will be considered as a violation of academic integrity. Please note that the former condition also holds for the material found on the web as everything on the web has been written by someone else. In short, using code from your peers or from any online sources as your own work **will not be tolerated**.

# Course Assignments Late Policy: 

You may use up to 7 grace days (in total) over the course of the semester. That is, you can submit your solutions without any penalty if you have free grace days left. Any additional unapproved late submission will be punished (1 day late: 20% off, 2 days late: 40% off, 3 days late: 50% off) and no submission after 3 days will be accepted.

You can use 3 grace days per assignment at most. 

For example, if you submit your Assignment-1 4 days late; your 3 grace days will be used and you'll be punished with 20% off because of +1. If you submit your Assignment-1 7 days late; again your 3 grace days will be used and we'll not accept your submission because of +4.


# Submission

1. Save your notebook and clear all the cells: Select Cell -> All Output -> Clear. This will clear all the outputs from all cells (but will keep the content of all cells).
2. Select Cell -> Run All. This will run all the cells in order, and will take several minutes.
3. Once you've rerun everything, select File -> Download as -> PDF via LaTeX (If you have trouble using "PDF via LaTex", you can also save the webpage as pdf. Make sure all your solutions especially the coding parts are displayed in the pdf, it's okay if the provided codes get cut off because lines are not wrapped in code cells).
4. Look at the PDF file and make sure all your solutions are there, displayed correctly. Name your pdf as **username_assignment1.pdf**, e.g. eacikgoz_assignment1.pdf.
5. Download a .ipynb version of your notebook, please name this as **username_assignment1.ipynb**, e.g. eacikgoz17_assignment1.ipynb.
6. You must submit both of your **username_assignment1.pdf** and **username_assignment1.ipynb** files to the Blackboard, before the deadline. Please note that failing to submit a PDF version of your Jupyter notebook, by following the steps above, will result in a deduction of points.
7. Finally, submit your overleaf report in PDF format as **username_assignment1_report.pdf**.

At the end, you msut have three different deliverables from Blackboard: (i) your jupyter notebook, (ii) PDF version of your jupyter notebook, (iii) PDF version of your overleaf report.
