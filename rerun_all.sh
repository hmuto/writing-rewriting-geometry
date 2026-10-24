#!/bin/sh
# Rerun every analysis of the paper, in order.
#   sh rerun_all.sh               start from the shipped sentence tables
#   RAW=1 sh rerun_all.sh         also rerun steps that need the raw CoAuthor logs
#   API=1 sh rerun_all.sh         also rerun steps that call the OpenAI API
# Without API=1, OpenAI embeddings must already be in results/*/emb_openai.npy.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
L=results/logs
mkdir -p $L
step () { echo "=== $1"; shift; "$@"; }
both () { name=$1; shift; for e in st openai; do step "$name $e" $PY "$@" $e > $L/$(echo "$name" | tr ' ' '_')_$e.log 2>&1; done; }

# Extraction (raw logs)
if [ "$RAW" = 1 ]; then
  step "replay extraction" $PY replay_extract.py > $L/replay_extract.log 2>&1
  step "menu ids"          $PY menu_ids.py       > $L/menu_ids.log 2>&1
  step "pool counts"       $PY pool_counts.py    > $L/pool_counts.log 2>&1
  step "process flags"     $PY process_flags.py  > $L/process_flags.log 2>&1
  step "rewriting subtypes" $PY mixed_subtypes.py > $L/mixed_subtypes.log 2>&1
  step "original suggestions" $PY mixed_origins.py extract > $L/origins_extract.log 2>&1
  step "accepted menus"    $PY sibling_free.py parse > $L/sibfree_parse.log 2>&1
  step "position meta"     $PY position_control.py build > $L/position_build.log 2>&1
  step "legacy extraction" $PY legacy_extraction.py > $L/legacy_extraction.log 2>&1
fi

# Embeddings
step "embed st" $PY fullscale_embed.py st replay > $L/embed_st.log 2>&1
if [ "$API" = 1 ]; then
  step "embed openai" $PY fullscale_embed.py openai replay > $L/embed_openai.log 2>&1
fi

# Main comparisons (Results)
for e in st openai; do
  step "forms $e"          $PY geometry_forms.py $e replay              > $L/forms_$e.log 2>&1
  step "forms lenmatch $e" $PY geometry_forms.py $e replay --lenmatch   > $L/forms_lenmatch_$e.log 2>&1
  step "forms lenmatch 3-set $e" $PY geometry_forms.py $e replay --lenmatch --nomixed > $L/forms_lenmatch_nomixed_$e.log 2>&1
  step "forms fullsent $e" $PY geometry_forms.py $e replay --fullsent   > $L/forms_fullsent_$e.log 2>&1
  step "forms menucap $e"  $PY geometry_forms.py $e replay --menucap    > $L/forms_menucap_$e.log 2>&1
  step "dose $e"           $PY geometry_dose.py $e replay               > $L/dose_$e.log 2>&1
  step "dose fullsent $e"  $PY geometry_dose.py $e replay --fullsent    > $L/dose_fullsent_$e.log 2>&1
done
step "forms decorrelate st" $PY geometry_forms.py st replay --decorrelate > $L/forms_dec_st.log 2>&1
step "forms lenmatch bins"  $PY geometry_forms.py st replay --lenmatch --binw 6 > $L/forms_bw6.log 2>&1
step "forms lenmatch bins"  $PY geometry_forms.py st replay --lenmatch --binw 8 > $L/forms_bw8.log 2>&1
both "orientation"      spectral_null.py
both "dose orientation" dose_orientation.py

# Scalar benchmark (S.III) and radius (S.IV)
for e in st openai; do
  step "scalar $e"          $PY fullscale_analyze.py $e replay      > $L/scalar_$e.log 2>&1
  step "scalar variants $e" $PY scalar_sensitivity.py $e replay    > $L/scalar_variants_$e.log 2>&1
done
both "radius sweep" radius_sweep.py

# Checks of the writing-rewriting contrast (S.II)
both "same sessions"    session_matched.py
both "subtype geometry" mixed_subtype_geometry.py
if [ "$API" = 1 ]; then step "origins embed openai" $PY mixed_origins.py embed openai > $L/origins_embed_openai.log 2>&1; fi
step "origins embed st" $PY mixed_origins.py embed st > $L/origins_embed_st.log 2>&1
for e in st openai; do
  step "origins $e"        $PY mixed_origins.py analyze $e          > $L/origins_$e.log 2>&1
  step "origins single $e" $PY mixed_origins.py analyze $e --single > $L/origins_single_$e.log 2>&1
done
both "corrected PR"        pr_bias_corrected.py
both "nonlinear dimension" nonlinear_dimension.py
for e in st openai; do
  step "union $e"          $PY union_dimension.py main $e            > $L/union_$e.log 2>&1
  step "union centred $e"  $PY union_dimension.py main $e --centered > $L/union_centered_$e.log 2>&1
done

# Robustness inventory (S.V)
both "null calibration"       null_calibration.py
both "writer mixing"          check_writer_mixing.py
both "all controls"           check_all_controls.py
both "within-session null"    check_within_session_null.py
for e in st openai; do step "stacked null $e" $PY check_stacked_null.py $e > $L/stacked_null_$e.log 2>&1; done
both "near duplicates"        near_duplicates.py
both "sibling-free"           sibling_free.py
both "menu matched"           menu_matched.py
step "upper bounds"           $PY upper_bounds.py > $L/upper_bounds.log 2>&1
both "orientation by k"       orient_k_sweep.py
both "principal angles"       principal_angles.py
both "sentence regression"    sentence_regression.py
both "session span and Vendi" session_span_vendi.py
both "rewrite clustering"     rewrite_clustering.py
# Replicate counts as used in the paper.
step "writer bootstrap st"     $PY writer_bootstrap.py st 300     > $L/writer_boot_st.log 2>&1
step "writer bootstrap openai" $PY writer_bootstrap.py openai 150 > $L/writer_boot_openai.log 2>&1
step "writer-prompt bootstrap st"     $PY writer_prompt_bootstrap.py st 200     > $L/wp_boot_st.log 2>&1
step "writer-prompt bootstrap openai" $PY writer_prompt_bootstrap.py openai 100 > $L/wp_boot_openai.log 2>&1
for e in st openai; do
  for f in --nodup --wdistinct "--vs-mixed" "--wdistinct --vs-mixed"; do
    step "writer-prompt bootstrap $e $f" $PY writer_prompt_bootstrap.py $e 1000 $f > $L/wp_boot_$e$(echo "$f" | tr -d ' -').log 2>&1
  done
done
step "standard bootstrap" $PY standard_bootstrap.py . results/replay/standard_bootstrap 2000 > $L/standard_bootstrap.log 2>&1
both "seed sweep" seed_sweep.py
step "T5 embeddings"    $PY other_encoders.py t5   > $L/emb_t5.log 2>&1
step "GPT-2 embeddings" $PY other_encoders.py gpt2 > $L/emb_gpt2.log 2>&1
step "forms T5"         $PY geometry_forms.py t5   replay > $L/forms_t5.log 2>&1
step "forms GPT-2"      $PY geometry_forms.py gpt2 replay > $L/forms_gpt2.log 2>&1
if [ "$API" = 1 ]; then
  step "paraphrases"      $PY paraphrase_control.py para  > $L/paraphrase_para.log 2>&1
  step "paraphrase embed" $PY paraphrase_control.py embed > $L/paraphrase_embed.log 2>&1
fi
for e in st openai; do step "paraphrase $e" $PY paraphrase_control.py analyze $e > $L/paraphrase_$e.log 2>&1; done
step "supporting statistics" $PY misc_stats.py > $L/misc_stats.log 2>&1

# Session thirds (S.VI), residual spectrum (S.VII), moderators (S.VIII)
both "temporal" geometry_temporal.py
for e in st openai; do step "temporal orientation $e" $PY orientation_frames.py temporal $e > $L/temporal_orient_$e.log 2>&1; done
step "residual spectrum" $PY residual_spectrum.py openai > $L/residual_spectrum.log 2>&1
both "moderators" geometry_moderator.py

# GPT-5 reference (S.IX)
if [ "$API" = 1 ]; then
  step "GPT-5 pools" $PY gpt5_regen.py > $L/gpt5_regen.log 2>&1
  step "GPT-5 build" $PY gpt5_analyze.py build > $L/gpt5_build.log 2>&1
  step "GPT-5 embed openai" $PY fullscale_embed.py openai gpt5ref > $L/gpt5_embed_openai.log 2>&1
fi
step "GPT-5 embed st" $PY fullscale_embed.py st gpt5ref > $L/gpt5_embed_st.log 2>&1
for e in st openai; do
  step "GPT-5 $e"             $PY gpt5_analyze.py $e              > $L/gpt5_$e.log 2>&1
  step "GPT-5 pool match $e"  $PY pool_lenmatch.py $e             > $L/gpt5_lenmatch_$e.log 2>&1
  step "GPT-5 orientation $e" $PY orientation_frames.py gpt5 $e   > $L/gpt5_orient_$e.log 2>&1
  step "GPT-5 union $e"       $PY union_dimension.py gpt5 $e      > $L/gpt5_union_$e.log 2>&1
done

# Writing process and proxy matching (S.X)
for e in st openai; do
  step "process delta $e"    $PY process_delta.py $e 0.30         > $L/process_delta30_$e.log 2>&1
  step "process delta 20 $e" $PY process_delta.py $e 0.20         > $L/process_delta20_$e.log 2>&1
  step "process robust $e"   $PY process_robust.py $e             > $L/process_robust_$e.log 2>&1
  step "process robust lm $e" $PY process_robust.py $e --lenmatch > $L/process_robust_lm_$e.log 2>&1
  step "coherence $e"        $PY coherence_check.py $e            > $L/coherence_$e.log 2>&1
  step "position $e"         $PY position_control.py $e           > $L/position_$e.log 2>&1
  step "selection h=0 $e"    $PY selection_h0.py $e               > $L/selection_h0_$e.log 2>&1
  step "pure h=1 $e"         $PY orientation_frames.py pure $e    > $L/pure_$e.log 2>&1
done

# Fixed-design and session-contribution checks (S.XI)
step "sampling sensitivity" $PY sampling_sensitivity.py . results/sampling_sensitivity > $L/sampling_sensitivity.log 2>&1

# Legacy direct extraction (S.I, S.III)
if [ "$RAW" = 1 ]; then
  step "legacy embed st" $PY fullscale_embed.py st fullscale > $L/legacy_embed_st.log 2>&1
  if [ "$API" = 1 ]; then step "legacy embed openai" $PY fullscale_embed.py openai fullscale > $L/legacy_embed_openai.log 2>&1; fi
  for e in st openai; do step "legacy scalar $e" $PY fullscale_analyze.py $e fullscale > $L/legacy_scalar_$e.log 2>&1; done
fi

# Figures and summary
step "summary of W-R checks" $PY wr_checks_summary.py > $L/wr_summary.log 2>&1
step "paper figures"         $PY make_paper_figures.py > $L/paper_figures.log 2>&1
step "supplementary figures" $PY make_figures.py       > $L/figures.log 2>&1
echo "=== done"
