from mayo.override.base import ChainOverrider
from mayo.override.quantize import (
    ThresholdBinarizer, FixedPointQuantizer,
    CourbariauxQuantizer, DGQuantizer, DGTrainableQuantizer,
    FloatingPointQuantizer, ShiftQuantizer, LogQuantizer,
    MayoDynamicFixedPointQuantizer, MayoFixedPointQuantizer,
    MayoRecentralizedFixedPointQuantizer)
from mayo.override.prune import (
    ThresholdPruner, MeanStdPruner, DynamicNetworkSurgeryPruner,
    MayoDNSPruner)


FPQuantizer = FloatingPointQuantizer
DNSPruner = DynamicNetworkSurgeryPruner
MayoRFPQuantizer = MayoRecentralizedFixedPointQuantizer
MayoDFPQuantizer = MayoDynamicFixedPointQuantizer
__all__ = [
    ChainOverrider,
    ThresholdBinarizer, FixedPointQuantizer,
    CourbariauxQuantizer, DGQuantizer, DGTrainableQuantizer,
    FloatingPointQuantizer, FPQuantizer, ShiftQuantizer, LogQuantizer,
    ThresholdPruner, MeanStdPruner, DynamicNetworkSurgeryPruner, DNSPruner,
    MayoDNSPruner, MayoDynamicFixedPointQuantizer,
    MayoDFPQuantizer, MayoFixedPointQuantizer,
    MayoRecentralizedFixedPointQuantizer, MayoRFPQuantizer,
]
