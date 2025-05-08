#python 
# call program : /home/blinkdrive/Documents/Projects/Summer2024/motion-diffusion-model/utils/pymesh.py 

# Input object[SMPL] : /home/blinkdrive/Documents/Projects/Summer2024/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_example_text_prompts/sample03_rep00_obj/ 

#Output folder : /home/blinkdrive/Documents/Projects/Summer2024/motion-diffusion-model/save/images/



import pymeshlab
import trimesh
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid Qt conflicts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import imageio.v2 as imageio  # Updated to handle deprecation warning
from tqdm import tqdm

def rotate_and_save_frame(obj_file_path, output_image_path):
    # Create a new MeshSet
    ms = pymeshlab.MeshSet()

    # Load the .obj file into the MeshSet
    ms.load_new_mesh(obj_file_path)

    # Extract the mesh
    mesh = ms.current_mesh()

    # Convert to Trimesh for visualization
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Plot the mesh using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Rotate the view to face forward with custom angles for front view
    ax.view_init(elev=0, azim=0)  # Adjust these values for a custom view angle

    ax.plot_trisurf(tri_mesh.vertices[:, 0], tri_mesh.vertices[:, 1], tri_mesh.vertices[:, 2], triangles=tri_mesh.faces, cmap='viridis', lw=1)
    plt.axis('off')

    # Save the plot as an image file
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_images_from_folder(input_folder, output_image_folder):
    obj_files = sorted([file_name for file_name in os.listdir(input_folder) if file_name.endswith('.obj')])
    total_files = len(obj_files)

    # Create the output folder if it does not exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    for idx, file_name in enumerate(tqdm(obj_files, desc="Processing files", unit="file")):
        obj_file_path = os.path.join(input_folder, file_name)
        output_image_path = os.path.join(output_image_folder, f'frame{idx}.png')
        rotate_and_save_frame(obj_file_path, output_image_path)

    print(f"Images saved to {output_image_folder}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python pymesh.py <input_folder> <output_image_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_image_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        sys.exit(1)

    create_images_from_folder(input_folder, output_image_folder)

if __name__ == "__main__":
    main()

