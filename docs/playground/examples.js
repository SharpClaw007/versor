// Example programs for the playground, in .vasm.
export const EXAMPLES = {
  countdown: `; countdown: a loop is a cycle in the chain graph
.name countdown

.chain entry
        LOADI 1
        MOVR r0                          ; R0 = unit decrement
        LOADI 5                          ; A = counter
loop:   OUT
        SUB r0
        BR -x: HALT -> end, +x: NOP -> loop
        ; exit listed first: wins the tie when A hits (0,0,0)
`,

  add_two: `; orientation is the argument: the same chain, called twice,
; sweeps different displacements after a pi frame rotation
.name add_two

.chain main
        LOADI 1          ; A = world +x, the branch input for both calls
        CALL fork
        OUT              ; prints 0.6
        ROTH pi          ; segments after this are re-aimed automatically
        CALL fork
        OUT              ; prints -2.5
        HALT

.chain fork
        BR (1,0,0): LOADI 0.6 -> a, (-1,0,0): LOADI 2.5 -> b
`,

  helix: `; the frame IS the geometry: identical frame-local laps,
; corkscrewed through world space by an accumulating ROTG
.name helix

.chain entry
        LOADI 2
        MOVR r0
        ROTG pi/4
        LOADI 2
        MOVR r0
        ROTG pi/4
        LOADI 2
        MOVR r0
        ROTG pi/4
        LOADI 2
        MOVR r0
        ROTG pi/4
        LOADI 2
        MOVR r0
        ROTG pi/4
        LOADI 2
        MOVR r0
        ROTG pi/4
        LOADI 2
        MOVR r0
        ROTG pi/4
        LOADI 2
        MOVR r0
        ROTG pi/4
        HALT
`,

  fib: `; iterative Fibonacci: R0=a, R1=b, R2=scratch, R3=counter.
; the counter's decrement unit is minted each lap by NORMing it.
.name fib

.chain entry
        LOADI 1
        MOVR r1
        LOADI 8
        MOVR r3
loop:   MOVA r0
        ADD r1
        MOVR r2
        MOVA r1
        MOVR r0
        MOVA r2
        MOVR r1
        OUT
        MOVA r3
        NORM
        MOVR r2
        MOVA r3
        SUB r2
        MOVR r3
        BR -x: HALT -> end, +x: NOP -> loop
`,

  hello: `; OUT in char mode; the program is the skyline of its char codes,
; turned 90 degrees per letter into a rectangular spiral
.name hello

.chain entry
        LOADI 72
        OUTC
        ROTH pi/2
        LOADI 101
        OUTC
        ROTH pi/2
        LOADI 108
        OUTC
        ROTH pi/2
        LOADI 108
        OUTC
        ROTH pi/2
        LOADI 111
        OUTC
        ROTH pi/2
        LOADI 44
        OUTC
        ROTH pi/2
        LOADI 32
        OUTC
        ROTH pi/2
        LOADI 119
        OUTC
        ROTH pi/2
        LOADI 111
        OUTC
        ROTH pi/2
        LOADI 114
        OUTC
        ROTH pi/2
        LOADI 108
        OUTC
        ROTH pi/2
        LOADI 100
        OUTC
        ROTH pi/2
        LOADI 33
        OUTC
        ROTH pi/2
        LOADI 10
        OUTC
        HALT
`,

  memory: `; memory is space: store at a cell, walk away, come back by a
; different route, and the value is still standing there
.name memory

.chain entry
        LOADI 7          ; A = 7, machine walks to (7,0,0)
        STORE 0.5        ; arrive (6.5,0,0): M[(6,0,0)] = A
        OP MOVR 2.0      ; walk away: (6.5,-2,0)
        LOADI 3          ; further: (9.5,-2,0), A clobbered
        OP DOT 1.0606601717798212     ; zig: (8.75,-2.75,0)
        OP SCALE 3.181980515339464    ; zag: (6.5,-0.5,0)
        LOAD 1           ; land inside cell (6,0,0): A = 7 again
        OUT
        HALT
`,
};
