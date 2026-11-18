# Show stderr and the tail of outputs
Hypothesis: failures (build-pov-ray, financial-document-processor) lost information: stderr was discarded and long outputs were cut to the first 1000 chars, so compiler/linker errors (printed last) and tool diagnostics were hidden and the model guessed or looped.
Change: commands run with 2>&1; output limit 1000 -> 3000 chars, and over-long output keeps head and tail with a "chars cut" marker.
Expected: fewer blind retries on build/tool errors, faster fixes; could cost more input tokens per step and add some stderr noise.