"""Contains dictionaries to be read by icecube.ml_suite.EventFeatureFactory
"""

feature_configs=dict()

# @TODO: add LLP feature config

feature_configs["mod_harnisch"]=\
"""
pulse_key: PULSEKEY_PLACEHOLDER

dom_exclusions:
  exclusions:
  - SaturationWindows
  - BadDomsList
  - CalibrationErrata
  partial_exclusion: false

# Define pulse modifier to use. Leave empty if pulses are not modified
pulse_modifier:
  class: ChargeWeightedMeanTimePulseModifier
  kwargs: {}

feature_config:
    features:
    - class: TotalCharge
      kwargs: {}
    - class: ChargeUntilT
      kwargs:
        times: [10, 50, 100]
    - class: TSpread
    - class: ChargeWeightedStd
    - class: ChargeWeightedMean
    - class: TFirstPulse
      kwargs: {}
    - class: TimeAtChargePercentile
      kwargs:
        percentiles: [0.01, 0.03, 0.05, 0.11, 0.15, 0.2, 0.5, 0.8]
"""

feature_configs["dnn_paper"]=\
"""
pulse_key: PULSEKEY_PLACEHOLDER

dom_exclusions:
  exclusions:
  - SaturationWindows
  - BadDomsList
  - CalibrationErrata
  partial_exclusion: false

# Define pulse modifier to use. Leave empty if pulses are not modified
pulse_modifier:
  class: ChargeWeightedMeanTimePulseModifier
  kwargs: {}

feature_config:
    features:
    - class: TotalCharge
      kwargs: {}
    - class: ChargeUntilT
      kwargs:
        times: [100, 500]
    - class: ChargeWeightedStd
    - class: ChargeWeightedMean
    - class: TFirstPulse
      kwargs: {}
    - class: TLastPulse
      kwargs: {}
    - class: TimeAtChargePercentile
      kwargs:
        percentiles: [0.2,0.5]
"""