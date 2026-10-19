"""Only bind real warmed Quant inputs needed by calibration; no evidence chain."""
import inspect

def bind_actual_launch(receiver,frame,encode=None):
    launcher=receiver.launchers[0]
    bound=inspect.signature(launcher).bind(*frame.f_locals['args'],**frame.f_locals.get('kwargs',{}),grid=frame.f_locals['grid'],stream=frame.f_locals['stream'])
    values=dict(launcher.config.kwargs)
    values.update({k:v for k,v in bound.arguments.items() if k not in ['grid','stream']})
    return values,None
