# Week 2 ESC-50 2-mix Validation Report

- output_root: `D:\evado_onedrive\OneDrive\Desktop\通识公选课\深度学习入门\深度学习大作业\week2\data_esc50_2mix`
- dataset: `ESC-50`
- sample_rate: `16000`
- duration: `5.0`
- selected_categories: `dog, rain, chirping_birds, keyboard_typing, door_wood_knock, clock_tick, glass_breaking, vacuum_cleaner, siren, car_horn`

## Summary

- clean source count: 400
- mixture count: train=10000, val=1000, test=1000, total=12000
- manifest records: train=20000, val=2000, test=2000, total=24000
- additivity check: PASS (max_abs_error=0.00006104, tolerance=0.0002)
- clipping check: PASS (max_peak=0.989960, limit=0.99)
- shape/sample-rate check: PASS
- split leakage check: PASS
- compatibility check: PASS

## Clean Source Counts

```text
split | clean_sources
------+--------------
train | 240          
val   | 80           
test  | 80           
```

```text
split | category        | count
------+-----------------+------
test  | car_horn        | 8    
test  | chirping_birds  | 8    
test  | clock_tick      | 8    
test  | dog             | 8    
test  | door_wood_knock | 8    
test  | glass_breaking  | 8    
test  | keyboard_typing | 8    
test  | rain            | 8    
test  | siren           | 8    
test  | vacuum_cleaner  | 8    
train | car_horn        | 24   
train | chirping_birds  | 24   
train | clock_tick      | 24   
train | dog             | 24   
train | door_wood_knock | 24   
train | glass_breaking  | 24   
train | keyboard_typing | 24   
train | rain            | 24   
train | siren           | 24   
train | vacuum_cleaner  | 24   
val   | car_horn        | 8    
val   | chirping_birds  | 8    
val   | clock_tick      | 8    
val   | dog             | 8    
val   | door_wood_knock | 8    
val   | glass_breaking  | 8    
val   | keyboard_typing | 8    
val   | rain            | 8    
val   | siren           | 8    
val   | vacuum_cleaner  | 8    
```

## Target Counts

```text
split | target_label    | count
------+-----------------+------
test  | car_horn        | 201  
test  | chirping_birds  | 201  
test  | clock_tick      | 200  
test  | dog             | 200  
test  | door_wood_knock | 200  
test  | glass_breaking  | 200  
test  | keyboard_typing | 200  
test  | rain            | 201  
test  | siren           | 200  
test  | vacuum_cleaner  | 197  
train | car_horn        | 1999 
train | chirping_birds  | 1999 
train | clock_tick      | 1999 
train | dog             | 2000 
train | door_wood_knock | 2001 
train | glass_breaking  | 2000 
train | keyboard_typing | 2001 
train | rain            | 2002 
train | siren           | 2000 
train | vacuum_cleaner  | 1999 
val   | car_horn        | 200  
val   | chirping_birds  | 200  
val   | clock_tick      | 202  
val   | dog             | 200  
val   | door_wood_knock | 201  
val   | glass_breaking  | 199  
val   | keyboard_typing | 200  
val   | rain            | 200  
val   | siren           | 199  
val   | vacuum_cleaner  | 199  
```

## Audio Statistics

- checked_mixtures: 12000
- max_abs_error: 0.00006104
- max_mean_abs_error: 0.00002139
- max_peak: 0.989960
- mixture_rms: min=0.022094, mean=0.109145, max=0.204490
- stem_rms: min=0.008823, mean=0.073876, max=0.177705

## Figures

- `figures/class_count_bar.png`
- `figures/target_count_bar.png`
- `figures/example_stem_0_spectrogram.png`
- `figures/example_stem_1_spectrogram.png`
- `figures/example_mixture_spectrogram.png`
