# Homework 3: GANs & Diffusion Models

**Due May 12, 11:59pm on Blackboard**

In this homework, you will implement and train some GANs as well as diffusion models. The homework consists of two components:

* `assignment3.ipynb` : Contains all of the coding questions, and will automatically generate and display results for you after completing each question. Open it on Colab by clicking on the file, and then "Open in Colab" at the top. Make sure that you save a copy of the notebook on your own drive to keep your progress. On Colab, download the result images from the left dashboard (folder icon). You will use these images in your report PDF. **Submit a PDF version of the notebook to the code (Print -> Preview -> Save) on BB in the assignment with (code)**

* `hw3_latex` :  Contains LaTeX files and figures needed to generate your PDF submission. Copy the images saved from the notebook into the `figures/` folder and fill out the empty test losses. Run `assignment3_latex.tex` to generate your PDF. **Submit the Latex PDF in the assignment with (PDF)**

You can open the notebook in Google Colab to get access to a free GPU, or you can link Colab to a local runtime to run it on your own GPU. 

# Assignment Policy

In our Deep Unsupervised Learning course, all work on assignments must be done individually unless stated otherwise. You can make discussions in an "abstract" way. However, it's crucial to distinguish between constructive discussions and the inappropriate use of resources: turning in someone else’s or your friend's work, in whole or in part, as your own will be considered as a violation of academic integrity. Please note that the former condition also holds for the material found on the web as everything on the web has been written by someone else. In short, using code from your peers or from any online sources as your own work **will not be tolerated**.

# Course Assignments Late Policy: 

You may use up to 7 grace days (in total) over the course of the semester. That is, you can submit your solutions without any penalty if you have free grace days left. Any additional unapproved late submission will be punished (1 day late: 20% off, 2 days late: 40% off, 3 days late: 50% off) and no submission after 3 days will be accepted.

You can use 3 grace days per assignment at most. 

For example, if you submit your Assignment-1 4 days late; your 3 grace days will be used and you'll be punished with 20% off because of +1. If you submit your Assignment-1 7 days late; again your 3 grace days will be used and we'll not accept your submission because of +4.


# Table of Contents
* *Question1: DCGAN on Emoji Dataset (25 Points)*
* *Question 2: CycleGAN (25 Points)*
* *Question 3: Diffusion-based Generative Models (50 Points)*


# Submission

1. Save your notebook and clear all the cells: Select Cell -> All Output -> Clear. This will clear all the outputs from all cells (but will keep the content of all cells).
2. Select Cell -> Run All. This will run all the cells in order and will take several minutes.
3. Once you've rerun everything, select File -> Download as -> PDF via LaTeX (If you have trouble using "PDF via LaTex", you can also save the webpage as pdf. Make sure all your solutions especially the coding parts are displayed in the pdf, it's okay if the provided codes get cut off because lines are not wrapped in code cells).
4. Look at the PDF file and make sure all your solutions are there, and displayed correctly. Name your pdf as **username_assignment2.pdf**, e.g. eacikgoz_assignment2.pdf.
5. Download a .ipynb version of your notebook, please name this as **username_assignment2.ipynb**, e.g. eacikgoz17_assignment2.ipynb.
6. You must submit both of your **username_assignment2.pdf** and **username_assignment2.ipynb** files to the Blackboard, before the deadline. Please note that failing to submit a PDF version of your Jupyter notebook, by following the steps above, will result in a deduction of points.
7. Finally, submit your overleaf report in PDF format as **username_assignment2_report.pdf**.
8. Never submit `.zip` files!

At the end, you must have three different deliverables from Blackboard: (i) your jupyter notebook, (ii) PDF version of your jupyter notebook, (iii) PDF version of your overleaf report. Do not submit `.zip` files!!!