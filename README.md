# Running the Paper Examples

Run each command from the corresponding algorithm directory.

## RECTANGLE

```text
# D.1
python run.py --case d1

# D.1, key-only
python run.py --case d1 --mode key

# D.1, joint
python run.py --case d1 --mode joint

# D.2
python run.py --case d2

# D.2, key-only
python run.py --case d2 --mode key

# D.2, joint
python run.py --case d2 --mode joint
```

## Blink

```text
# D.3
python run.py --case d3

# D.4, key-only
python run.py --case d4 --mode key

# D.4, joint
python run.py --case d4 --mode joint

# D.5, key-only
python run.py --case d5 --mode key

# D.5, joint
python run.py --case d5 --mode joint
```

## GIFT-64

```text
# D.6
python run.py --case d6

# D.7
python run.py --case d7

# D.8
python run.py --case d8

# D.9
python run.py --case d9

# D.10
python run.py --case d10

# D.11
python run.py --case d11

# D.12
python run.py --case d12

# D.13
python run.py --case d13

# D.14
python run.py --case d14

# D.15
python run.py --case d15

# D.16
python run.py --case d16

# D.17
python run.py --case d17
```

To run only the key-only or joint analysis:

```text
python run.py --case d6 --mode key
python run.py --case d6 --mode joint
```

Replace `d6` with any case from `d6` through `d17`.

## SKINNY

```text
# D.18
bash run.sh d18

# D.19
bash run.sh d19

# D.20
bash run.sh d20

# D.21
bash run.sh d21

# D.22
bash run.sh d22

# D.23
bash run.sh d23
```

## Ascon

Run the following commands from `ascon/`:

```text
# DL.1
python run.py --case d1

# DL.2, second order
python run.py --case d2

# DL.3
python run.py --case d3

# DL.4
python run.py --case d4

# DL.5
python run.py --case d5

# DL.6
python run.py --case d6

# DL.7
python run.py --case d7
```

Run DL.8 from `ascon/prefix_2rd/`:

```text
python generate_graphs.py
python run.py
```

## Xoodoo

```text
# DL.9
python run.py --case d9

# DL.10
python run.py --case d10

# DL.11
python run.py --case d11

# DL.12
python run.py --case d12

# DL.13
python run.py --case d13

# DL.14
python run.py --case d14

# DL.15
python run.py --case d15

# DL.16
python run.py --case d16
```
