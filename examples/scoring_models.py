from candidate_models.model_commitments import brain_translated_pool
from brainscore import score_model
import pandas as pd 



def process_score(score):
    center, error = score.sel(aggregation='center'), score.sel(aggregation='error')
    print(f"score: {center.values:.3f}+-{error.values:.3f}")
    return [center,error]

results = []
for identifier in brain_translated_pool:
    if identifier in ['xception','densenet-121','densenet-169','densenet-201']:continue
    model = brain_translated_pool[identifier]
    score_combined,error_combined = process_score(score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.MajajHong2015.IT-pls_combined'))
    score_tz_pos,error_tz_pos   =   process_score(score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.MajajHong2015.IT-pls_combined_split_tz_01_pos'))
    score_ty_pos,error_ty_pos   =   process_score(score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.MajajHong2015.IT-pls_combined_split_ty_01_pos'))
    score_tz_neg,error_tz_neg   =   process_score(score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.MajajHong2015.IT-pls_combined_split_tz_01_neg'))
    score_ty_neg,error_ty_neg   =   process_score(score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.MajajHong2015.IT-pls_combined_split_ty_01_neg'))
    
    results.append([identifier,
                    score_combined,error_combined,
                    score_tz_pos,error_tz_pos,
                    score_ty_pos,error_ty_pos,
                    score_tz_neg,error_tz_neg,
                    score_ty_neg,error_ty_neg,model.layers
                                ])
    df = pd.DataFrame(results,columns=['identifier','score combined','error combined',
                                   'score_tz_pos','error_tz_pos',
                                   'score_ty_pos','error_ty_pos',
                                   'score_tz_neg','error_tz_neg',
                                   'score_ty_neg','error_ty_neg','layers'])

    import pdb;pdb.set_trace()
    print(df)
    df.to_csv('results.csv')


df = pd.DataFrame(results,columns=['identifier','score combined','error combined',
                                   'score_tz_pos','error_tz_pos',
                                   'score_ty_pos','error_ty_pos',
                                   'score_tz_neg','error_tz_neg',
                                   'score_ty_neg','error_ty_neg'])

print(df)
df.to_csv('results.csv')
