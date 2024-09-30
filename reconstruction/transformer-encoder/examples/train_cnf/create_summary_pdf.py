import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse

def plot_directory(directory):
    # Add a title page for each directory
    fig = plt.figure()
    plt.text(0.5, 0.5, os.path.basename(directory), ha='center', va='center', size=24)
    plt.axis('off')  # Hide axes
    pdf.savefig(fig, dpi=300)
    
    # get all images
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    n_images = len(image_files)
    
    # 2x2 grid
    n_cols = 2
    n_rows = 2
    
    for i in range(0, n_images, n_cols*n_rows):
        # fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        fig, axs = plt.subplots(n_rows, n_cols)
        plt.tight_layout()
        for j in range(n_rows):
            for k in range(n_cols):
                idx = i + j*n_cols + k
                if idx >= n_images:
                    break
                img = mpimg.imread(os.path.join(directory, image_files[idx]))
                axs[j, k].imshow(img)
                axs[j, k].axis('off')
        pdf.savefig(fig, dpi=300)
        plt.close()
    
    # single plot per page
    # for file in os.listdir(directory):
    #     if file.endswith('.png'):
    #         # Create a new figure
    #         fig = plt.figure()
    #         # Load image
    #         img = mpimg.imread(os.path.join(directory, file))
    #         # Show image in figure
    #         plt.imshow(img)
    #         plt.axis('off')  # Hide axes
    #         # Save figure to PDF
    #         pdf.savefig(fig)
    #         plt.close()


# Create the argument parser
parser = argparse.ArgumentParser(description='Plot some events.')

parser.add_argument("-p", "--performance-dir", action="store",
    type=str, required=True, dest="performance-dir",
    help="Input performance folder.")
params = vars(parser.parse_args())  # dict()

# directory
topdir = params["performance-dir"]
directories = [os.path.join(topdir, d) for d in os.listdir(topdir) if os.path.isdir(os.path.join(topdir, d))]
print(directories)

# Create a new PDF file
pdf = PdfPages(f"{topdir}/summary.pdf")

datasetname = topdir.split("/")[-2]

# Add title slide
fig = plt.figure()
plt.text(0.5, 0.5, datasetname, ha='center', va='center', size=18)
plt.axis('off')  # Hide axes
pdf.savefig(fig)
plt.close()

# Add corner plot and performance histograms
fig = plt.figure()
img = mpimg.imread(f"{topdir}/performance_histograms.png")
plt.imshow(img)
plt.axis('off')  # Hide axes
pdf.savefig(fig, dpi=300)
plt.close()

fig = plt.figure()
img = mpimg.imread(f"{topdir}/corner_plot.png")
plt.imshow(img)
plt.axis('off')  # Hide axes
pdf.savefig(fig, dpi=300)
plt.close()

# plot input spectrum
plot_directory(f"{topdir}/input_spectrum")

# For each directory
for directory in directories:
    if "input_spectrum" in directory:
        continue
    plot_directory(directory)

# Close the PDF file
pdf.close()