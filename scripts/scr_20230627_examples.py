import sys as _sys

from sample import generate, edit
from visualize import render_mesh

def text_to_motion():
    _sys.argv.append('--model_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt')
    _sys.argv.append('--num_samples')
    _sys.argv.append('3')
    _sys.argv.append('--num_repetitions')
    _sys.argv.append('3')
    _sys.argv.append('--text_prompt')
    _sys.argv.append("the person walked forward and is picking up his toolbox.")
    generate.main()

def action_to_motion():
    _sys.argv.append('--model_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model/save/humanact12/model000350000.pt')
    _sys.argv.append('--num_samples')
    _sys.argv.append('3')
    _sys.argv.append('--num_repetitions')
    _sys.argv.append('3')
    _sys.argv.append('--action_name')
    _sys.argv.append("drink")
    generate.main()

def unconstrained_motion():
    _sys.argv.append('--model_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model/save/unconstrained/model000450000.pt')
    _sys.argv.append('--num_samples')
    _sys.argv.append('3')
    _sys.argv.append('--num_repetitions')
    _sys.argv.append('3')
    generate.main()

def render():
    _sys.argv.append('--input_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model\save\humanml_trans_enc_512\samples_humanml_trans_enc_512_000200000_seed10_the_person_kneeled_down\sample00_rep00.mp4')
    render_mesh.main()

def edit_in_between():
    _sys.argv.append('--model_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt')
    _sys.argv.append('--edit_mode')
    _sys.argv.append('in_between')
    edit.main()

if __name__ == "__main__":
    edit_in_between()