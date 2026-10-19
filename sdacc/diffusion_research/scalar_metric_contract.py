"""Official singleton ImageReward calculation, without unused legacy gates."""
import hashlib
import json
import math
import os
from pathlib import Path
IR_TOLERANCE = 2e-5

def official_singletons(reward, conditions, images, paths, torch_module, device, *, verify=True):
    """Official scalar once; independent audit only when explicitly requested."""
    assert len(conditions) == len(images) == len(paths)
    values, audit = [], []
    for condition, image, path in zip(conditions, images, paths):
        scalar = float(reward.score(condition['prompt'], str(path)))
        if not verify:
            assert math.isfinite(scalar)
            values.append(scalar)
            audit.append(dict(condition=condition['id'], official_scalar=scalar,
                independent_verification=False, official_calls=1))
            continue
        token = reward.blip.tokenizer(condition['prompt'], padding='max_length',
            truncation=True, max_length=35, return_tensors='pt').to(device)
        singleton = torch_module.stack([reward.preprocess(image)]).to(device)
        independent = float(reward.score_gard(token.input_ids, token.attention_mask,
            singleton).flatten().cpu().tolist()[0])
        difference = abs(scalar - independent)
        assert math.isfinite(scalar) and math.isfinite(independent)
        assert difference < IR_TOLERANCE, (condition['id'], scalar, independent, difference)
        values.append(scalar)
        audit.append(dict(condition=condition['id'], official_scalar=scalar,
            singleton_score_gard=independent, absolute_difference=difference,
            tolerance=IR_TOLERANCE, batch_size=1))
    return values, audit
