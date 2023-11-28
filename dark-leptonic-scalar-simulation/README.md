Test for simulating LLPs from scratch all the way to level 2.


### Generate MC data

1. Generate I3MCTree. Use llpgun.py. Also Polyplopia? Or add polyplopia in detector simulation i think.

2. Generate I3MCPESeriesMap. Use https://github.com/icecube/icetray/blob/main/simprod-scripts/resources/scripts/ppc.py
   Remeber to use GPUs for faster execution.

3. Generate detector response. Use https://github.com/icecube/icetray/blob/main/simprod-scripts/resources/scripts/detector.py

### Filtering
Use filterscripts or filterscripts_v2?

4. Simulate online filter (level 1). Remember to add LLPInfo to saved frames in SimulateFilter.py!

5. Simulate level 2.
