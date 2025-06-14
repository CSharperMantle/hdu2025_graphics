# hdu2025_graphics

Course work repository for HDU (2024-2025-2)-B0504060-05 "Computer Graphics".

## Contents

* [x] Interactive Bezier curve demo ([bezier/](bezier/))
* [x] 3D transformation demo ([transform/](transform/))
* [x] Project: Tetris 3D! ([tetris3d/](tetris3d/))

## Projects

### bezier

```sh
$ python ./bezier/bezier_interactive.py
```

* Left mouse button: Add a point to the current spline
* Middle mouse button: Finish the current spline and create another connected one
* Right mouse button: Remove last point from the current spline

<figure>
    <img src="assets/bezier_interactive.png" alt="interactive Bezier curve">
    <figcaption>Figure 1: Screenshot for interactive Bezier curve demo.</figcaption>
</figure>

### transform

```sh
$ python ./transform/transform.py
```

* `x`: Toggle between object and camera edit modes
* Object mode:
  * Left mouse drag: Rotate object
  * Right mouse drag: Move object
  * Mouse wheel: Scale object up/down
  * Space: Create a new object (add current to scene)
  * Backspace: Remove last object and make it active
  * `q`: Toggle between cube and sphere
  * `r`: Reset object transforms (rotation, scale, position)
  * `1`, `2`, `3`: Flip object symmetry along x, y, z axes
  * `c`: Change to a random color
  * `a`: Toggle object axes visibility
  * `,`: Create random object and add to scene
* Camera mode:
  * Left mouse drag: Orbit camera
  * Mouse wheel: Zoom in/out
  * `p`: Switch between perspective and parallel projections
* `ESC`: Exit program

<figure>
    <img src="assets/transform.png" alt="transformations demo">
    <figcaption>Figure 2: Screenshot for transformation demo.</figcaption>
</figure>

### tetris3d

```sh
$ python ./tetris3d/tetris3d.py
```

* Gameplay controls:
  * `w`, `a`, `s`, `d`: Move piece (direction relative to camera view)
  * `q`: Rotate piece around Y axis
  * `e`: Rotate piece around Z axis
  * `SPACE`: Drop piece instantly
  * Arrow keys: Pan camera
  * Left mouse drag: Rotate camera view
  * Mouse wheel: Zoom in/out
  * `HOME`: Reset camera position
  * `m`: Toggle grid marker visibility
  * `z`: Toggle block landing locator visibility
* Pause mode (press `` ` ``):
  * `` ` ``: Resume game
  * `1`, `2`, `3`: Select along X, Y, Z axis respectively
  * Right mouse click: "X-Ray on" on a layer along the axis
  * `m`: Toggle grid marker visibility
  * `z`: Toggle landing locator visibility
* `ESC`: Exit program

<figure>
    <img src="assets/tetris3d-1.png" alt="Tetris3D demo">
    <figcaption>Figure 3: Screenshot for Tetris3D.</figcaption>
</figure>

<figure>
    <img src="assets/tetris3d-xray.png" alt="Tetris3D X-Ray mode demo">
    <figcaption>Figure 4: X-Ray mode of Tetris3D.</figcaption>
</figure>

## License

### Program

Copyright &copy; 2025 Rong Bao <<webmaster@csmantle.top>>.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

[./LICENSE-Apache-2.0](./LICENSE-Apache-2.0) or <https://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an **"AS IS" BASIS**, **WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Documentation

Copyright &copy; 2025 Rong Bao <<webmaster@csmantle.top>>.

This work is licensed under Creative Commons Attribution-ShareAlike 4.0 International. To view a copy of this license, see [./LICENSE-CC-BY-SA-4.0](./LICENSE-CC-BY-SA-4.0) or visit <https://creativecommons.org/licenses/by-sa/4.0/>.
