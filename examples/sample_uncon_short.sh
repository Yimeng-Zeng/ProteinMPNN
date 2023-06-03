path_to_PDB="../inputs/PDB_monomers/rfdiff_uncon/2_5.pdb"

output_dir="../outputs/2_5_outputs_T0_01_N0_2A"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design="A"

python ../protein_mpnn_run.py \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
        --num_seq_per_target 10000 \
        --sampling_temp "0.01" \
        --seed 42 \
        --batch_size 100 \
        --ca_only True \
        --path_to_model_weights "../ca_model_weights"
