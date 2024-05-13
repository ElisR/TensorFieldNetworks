> This repository contains JAX code accompanying my internal talk titled "Representation Theory and $SO(3)$ in GNNs".

## 💬 Talk Notes

> [!NOTE]
>
> This is intended to be a not-too-technical introduction to representation theory, and only assumes basic linear algebra.


# Representation Theory of $SO(3)$

> [!WARNING]
>
> When reading up representation theory for $SO(3)$, be aware that most notation online is for a very closely-related group $SU(2)$.
> The translation back to $SO(3)$ notation is simple and involves removing complex numbers etc, but it can be slightly annoying so I have tried to stick to tangible $SO(3)$ quantities.

## ↪️ Rotations as a Group

Groups in mathematics can be quite abstract, so let’s just relate the rules for something to be a “group” to familiar rotations.

**Closure**. We are happy with the concept that if we rotate a real-life object around an axis, then rotate it around a different axis, we could have gotten the same end result through a single rotation about a single axis. [Proof: grab a nearby object - it will be useful for this entire section.]
That is, composing two rotations gives another rotation.
So if I construct a set that includes all possible rotations (and you forgive my loose language since there are uncountably many rotations), then under the binary operation of composition, two elements in my set give another element in my set.

**Identity**. Another intuitive fact is that there is only one rotation that does nothing: the “don’t bother” operation.
[Proof: try thinking of another one.]

**Inverse**. We are also happy that any rotation that we do has an inverse rotation that undoes that rotation, yielding the “don't bother” operation.
[Proof: play the footage of you rotating said object in reverse.]

**Associativity**. With these observations, we can almost say that what I have described above is a group, since a group requires a set of elements $G$ (here a set of rotations) and a binary operation $a \cdot b$ between elements $a, b$ belonging to $G$ (here composition).
I say “almost”, because a group also requires associativity: $a \cdot (b \cdot c) = (a \cdot b) \cdot c$, which the rotation group also has.
[Proof: trust me.]

A group is specified by the set of elements and a table specifying the result of applying the binary operation on every ordered pair of elements.

Note that one thing a group does not require is commutativity, meaning $a \cdot b \neq b \cdot a$, at least not necessarily.
Commutativity is something that rotations in 3D does not have (unlike 2D).
[Proof: grab an object, for me it is now a TV remote, then rotate it anti-clockwise along the x-axis, then anti-clockwise along the y-axis. If you swap the order of the rotations then the TV remote points in a different direction.]

The name of the group I have been describing is $SO(3)$, and is the focus of this article.

## 🔢 Representations

It turns out that one can satisfy the specification of any group by replacing group elements with certain (non-singular) matrices, and letting the binary operation be matrix multiplication.
The mapping from the set of elements to matrices is called a *representation*. (A point about language worth repeating to avoid confusion: the representation is the entire mapping from group elements to matrices - a single matrix is not a “representation” of a single group element, even though sounds like a reasonable English statement.)
For $\rho$ to be a representation, $\rho$ must respect the group structure (by being a so-called “homomorphism”):

$$
a \cdot b = c \implies \rho(a)\rho(b)=\rho(c)
$$

We have not yet specified the size of these matrices, because that depends on the exact details of the group at hand.

## 🔲 Irreducible Representations

In the land of representations, an important divide exists between those that are reducible, and the privileged few that are irreducible.

You may have noticed from the definitions above that if you have a representation, it is easy to construct another representation by stacking matrices along the block diagonal.
For example, the representation

$$
\rho_{1+2}(g) \equiv \begin{bmatrix}\rho_1(g) & 0 \\0 & \rho_2(g) \\\end{bmatrix}
$$

will satisfy the right group relations, provided $\rho_1$ and $\rho_2$ are valid representations, since the blocks never interact during matrix multiplication.
Furthermore, we can get more valid representations by doing a similarity transformation (change of basis) such as $\rho(g) \to \tilde{\rho}(g)=u \rho(g)u^{-1}$, which is still a homomorphism since $\tilde{\rho}(a)\tilde{\rho}(b)=u \rho(a)u^{-1}u\rho(b)u^{-1}=u\rho(a)\rho(b)u^{-1} = \tilde{\rho}(c)$ as required.
Hence there are plenty of ways to construct representations for free (i.e. without having to do pen and paper matrix multiplication) using the same starting representation.
This is why we distinguish between so-called “reducible” representations and “irreducible” representations (”irreps” for cool kids).

A representation is reducible if the matrices all group elements can *simultaneously* be brought to block diagonal form through a basis change.
(Simultaneous is an important word here, because the basis change has to be consistent across all group elements - otherwise each matrix in a representation could get its own basis change we would have a useless definition.)
Predictably, irreducible representations are representations that are not reducible: in some ways they are like “atoms” of a group.
(One could still have two different-looking irreps that are actually related by a change of basis, but we’d still call them both irreps.
Later on, we will avoid this confusion by picking a physically-motivated basis.)

## 🌼 Representations of SO(3)

If one has studied linear algebra in science or engineering, it is not surprising that the group of rotations can be represented by matrices, because one has already been exposed to the $3 \times 3$ rotation matrices that act on 3-dimensional vectors.
(Indeed, $SO(3)$ stands for the group of special orthogonal $3 \times 3$ matrices, so it better well be representable this way!)
However, we will now see that this is not the only possible representation of the group.

We know that we can represent an anti-clockwise rotation about the $z$-axis through angle $\theta$ by the block-diagonal matrix

$$
\boldsymbol{R}_z(\theta) =\left(\begin{array}{ccc}\cos \theta & -\sin \theta & 0 \\\sin \theta & \cos \theta & 0 \\0 & 0 & 1\end{array}\right),
$$

and likewise for the $x$ and $y$ axes by cyclic permutation of rows and columns.
We have chosen a basis such that this acts appropriately on familiar 3D vectors and their $x$, $y$ and $z$ coordinates.
We say that this base space is $\mathbb{R}^3$, and these matrices above represent the *group action* on this base space of positions $\mathbf{r} = (r_x, r_y, r_z)^T$.

One can verify that in fact all rotations about $x$, $y$ and $z$ can be generated through

$$
\boldsymbol{R}_i(\theta) = e^{\theta \boldsymbol{L}_i}, \qquad \text{where} \qquad e^{\boldsymbol{A}}=\sum_{k=0}^{\infty} \frac{1}{k !} \boldsymbol{A}^k=\boldsymbol{1} +\boldsymbol{A}+\frac{1}{2} \boldsymbol{A}^2+\cdots
$$

$$
\quad L_x = \left( \begin{array}{ccc}0 & 0 & 0 \\0 & 0 & -1 \\0 & 1 & 0\end{array} \right), \quad L_y = \left( \begin{array}{ccc}0 & 0 & 1 \\0 & 0 & 0 \\-1 & 0 & 0\end{array} \right), \quad L_z = \left( \begin{array}{ccc}0 & -1 & 0 \\1 & 0 & 0 \\0 & 0 & 0\end{array} \right)
$$

Notice also that these matrices are all antisymmetric, $L_i = - L_i^T$.
(As expected given that $\boldsymbol{R}_i(\theta) = e^{\theta \boldsymbol{L}_i}$ must be an orthogonal matrix since $(e^{\theta \boldsymbol{L}_i})^T=e^{-\theta \boldsymbol{L}_i}$ and hence $\boldsymbol{R}^T_i(\theta) \boldsymbol{R}_i(\theta) = e^{\mathbf{0}} = \mathbf{1}$.)
These $L_i$ are the *generators* of infinitesimal rotations.
Big rotations are composed of many infinitesimal rotations, so between them they generate all rotations.

Now for the cool bit, where we make the jump from mundane $3 \times 3$ rotation matrices.
Since we expanded $\boldsymbol{R}_i(\theta) = e^{\theta \boldsymbol{L}_i} = \boldsymbol{1} +\theta \boldsymbol{L}_i+\frac{1}{2} \theta^2 \boldsymbol{L}_i^2+\cdots$, it’s obvious that all that can matter in the group multiplication rules of $R_x(\alpha)$, $R_y(\beta)$, $R_z(\gamma)$ is that the algebraic relations of the matrices $\boldsymbol{L}_i$.
The algebra of the generators above is fully specified by the commutation relations

$$
L_x L_y - L_y L_x = L_z, \qquad L_y L_z - L_z L_y = L_x, \qquad L_z L_x - L_x L_z = L_y.
$$

Hence, if I have some other set of three larger matrices that also satisfy such commutation relations, they must also be generators in some larger representation!
(A proper derivation of why this is necessary might rely on some [less obvious](https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula) lemmas, but this motivates why such commutation relations would be sufficient to create a new representation.)

(One can also verify that the operator $L^2 = L_x^2 + L_y^2 + L_z^2$ consequently commutes with all the generators $L_i$.)

Funnily enough, these commutation relations can be satisfied by matrices of many shapes, not just $3 \times 3$, while still generating fresh new classes of irreps for $SO(3)$! For example, one can trivially set $L_x = L_y = L_z = (0)$, so that all rotation matrices are trivial $1 \times 1$ matrices $\boldsymbol{R}_i(\theta) = (1)$.
This is the representation that is relevant for scalars, in fact, which are invariant under rotation.
Taking it even further, there are compatible $5 \times 5$, $7 \times 7$ and $(2\ell + 1) \times (2 \ell + 1)$ irreducible representations for $\ell \in \mathbb{N}$.
The natural number $\ell$ that indexes the size of the irreducible representation is called the “angular momentum” (motivated for physical reasons).

So far, we have only discussed rotations about the Cartesian axes, rather than a generic axis.
One way to rotate around a generic axis $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]$ is $\boldsymbol{R}(\boldsymbol{\omega}) = \exp(\omega_x L_x + \omega_y L_y + \omega_z L_z)$, but another way to represent a rotation is with three three so-called “Euler angles” $\alpha, \beta, \gamma$:

$$
R(\alpha, \beta, \gamma) = e^{\alpha L_z} e^{\beta L_y} e^{\gamma L_z}.
$$

## ❓ How do Spherical Harmonics come in?

We are now ready to introduce spherical harmonics.
Thus far, the basis we were using for our matrices were the Cartesian axes, and each of these were on equal footing.
It is more convenient, however, to choose a single rotation axis as a reference.
We choose this to be the $z$ axis (but this choice is just a convention - the [e3nn](https://e3nn.org/) library chooses $y$, for example).

How do we make the $z$ axis special, you ask?
Well, after selecting the angular momentum $\ell$ (read: size) of our representation, we express rotation matrices in the eigenbasis of the $L_z$ operator.
That is, we define $2\ell + 1$ basis states $\boldsymbol{Y}^{\ell}_m(\mathbf{r})$ for $m \in \{ -\ell, -\ell + 1, \ldots , \ell -1, \ell\}$ to be given by

$$
\boldsymbol{L}_z \boldsymbol{Y}^{\ell}_m(\mathbf{r}) = -i m \boldsymbol{Y}^{\ell}_m(\mathbf{r}).
$$

Note that $\boldsymbol{Y}^{\ell}(\mathbf{r}): \mathbb{R}^{3} \mapsto \mathbb{R}^{2\ell + 1}$ is a function that takes in a point in 3D and outputs a single-column-array with $2\ell + 1$ elements.
Most of the functional form of $\boldsymbol{Y}^{\ell}_m(\mathbf{r})$ is still left undetermined by this equation, because $\boldsymbol{L}_z$ will just be a constant matrix, which we are trying to choose a basis for.
The basis for $\boldsymbol{L}_z$ gets fixed when we request the natural ordering of components to be

$$
\boldsymbol{Y}_{1}^{1}(\mathbf{r})=
\begin{bmatrix} 0 \\ 0 \\ Y_1^1(\mathbf{r})
\end{bmatrix}, \quad
\boldsymbol{Y}_{0}^{1}(\mathbf{r})=
\begin{bmatrix} 0 \\ Y_0^1(\mathbf{r}) \\ 0
\end{bmatrix}, \quad
\boldsymbol{Y}_{-1}^{1}(\mathbf{r}) = \begin{bmatrix} Y^1_{-1}(\mathbf{r}) \\ 0 \\ 0
\end{bmatrix}.
$$

Fixing this basis lets us instantiate the Wigner D-matrix, which is the representation $\mathbf{D}^{\ell}(R)$ for rotation group elements $R \in SO(3)$.
With the representations now concretely defined, the spherical harmonics are defined as the basis functions that transform appropriately under the orthogonal Wigner D-matrix:

$$
[\mathbf{D}^{\ell}(R)]^{-1} \boldsymbol{Y}^{\ell}(\mathbf{r}) = \boldsymbol{Y}^{\ell}(R \cdot \mathbf{r}),
$$

which fixes their functional form (to quite [unwieldy expressions](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics), at least according to my beauty standards).

(Choosing a different “special axis” from the start would just be an orthogonal transformation of the Wigner D-matrix, which would then modify the functional form of the spherical harmonics.)

### ❔Interlude: why is it $\boldsymbol{Y}(R \cdot  \mathbf{r}) = \boldsymbol{R}^{-1} \boldsymbol{Y}(\mathbf{r})$?

At first glance, this looks different to the equivariance relation for functions like GNNs.
There, GNNs like $\boldsymbol{F}(\mathbf{x})$ must satisfy $\boldsymbol{F}(\boldsymbol{R} \cdot  \mathbf{x}) = \boldsymbol{R} \boldsymbol{F}(\mathbf{x})$ to be equivariant.
Why the difference? The difference is that $\boldsymbol{Y}(\mathbf{r})$ is a signal on a base space—position space—whereas GNNs act on the *signals* $\mathbf{x}(\mathbf{r})$.
The only self-consistent way for a signal to behave under a group element the way it’s written above.

Why? A self-consistent definition of rotations on signals should satisfy $(\boldsymbol{S} (\boldsymbol{R} \boldsymbol{Y}))(\mathbf{r}) = (\boldsymbol{S} \boldsymbol{R}) \boldsymbol{Y}(\mathbf{r})$ for two rotations $\boldsymbol{S}$ and $\boldsymbol{R}$.
(First act on the signal with $\boldsymbol{R}$, then act with $\boldsymbol{S}$ and it should be the same as acting with $\boldsymbol{S} \boldsymbol{R}$ in one go since we act on the left of the signal.)
Let’s consider the alternative (incorrect) definition on signals $\boldsymbol{R} \boldsymbol{Y}(\mathbf{r}) = \boldsymbol{Y}(R \cdot \mathbf{r})$ and see if it would work.
Let $\boldsymbol{R}\boldsymbol{Y}(\mathbf{r})\equiv\tilde{\boldsymbol{Y}}(\mathbf{r})$ for clarity since that itself is a signal.
Applying our definition on the LHS we would have

$$
(\boldsymbol{S} (\boldsymbol{R} \boldsymbol{Y}))(\mathbf{r}) = \boldsymbol{S}\tilde{\boldsymbol{Y}}(\mathbf{r})=\tilde{\boldsymbol{Y}}(S \cdot\mathbf{r})=\boldsymbol{R}\boldsymbol{Y}(S \cdot\mathbf{r})=\boldsymbol{Y}(RS \cdot\mathbf{r}).
$$

Yet, on the RHS we should have $(\boldsymbol{S} \boldsymbol{R}) \boldsymbol{Y}(\mathbf{r})=\boldsymbol{Y}(SR \cdot \mathbf{r})$, and $RS \neq SR$!
Therefore, it wouldn’t work, but putting an inverse makes it work because $(SR)^{-1}=R^{-1}S^{-1}$.
So, $\boldsymbol{R} \boldsymbol{Y}(\mathbf{r}) = \boldsymbol{Y}(R^{-1} \cdot  \mathbf{r})$ it is.

One can also intuit this graphically, which I plan to add as a figure.

One final thing: Because rotations don’t affect the distance of a point in space to the origin, if we want the spherical harmonics to be easily normalisable, it’s better to treat them as functions of the spherical angles only, i.e. a function on the unit sphere $\mathbb{S}^2$, taking as input $\frac{\mathbf{r}}{|\mathbf{r}|}$.
Thus the function $\boldsymbol{Y}^{\ell}(\mathbf{r})$ is to be considered a map $\mathbb{S}^2 \to \mathbb{R}^{2\ell + 1}$.
(There are probably deeper reasons for this that I’m overlooking.)

## 🕺 Visualising Spherical Harmonics

In the same way that Fourier series basis functions become increasingly fine with higher momentum (e.g. $\sin(k \phi)$ with larger $k \in \mathbb{Z}$), the spherical harmonics get more intricate with higher angular momentum $\ell$.
However, unlike $\cos$ and $\sin$, spherical harmonics get much wackier! Below is a simple visualisation hosted on HuggingFace - try setting $\ell = 8$ below!
(Source code found [here](https://huggingface.co/spaces/ElisR/spherical_harmonics_visualisation).)

[Here](https://elisr-spherical-harmonics-visualisation.hf.space/), we plot a surface with its spherical polar coordinates satisfying

$$
r(\phi, \theta) \propto | \Re [Y_{\ell m}(\phi, \theta)] |,
$$

and the colour of the surface gives the sign of $\Re [Y_{\ell m}(\phi, \theta)]$, where I denote by $Y_{\ell m}(\phi, \theta)$ the complex spherical harmonics (more commonly found online).

(**Note**, the real “real spherical harmonics” are not simply the real components of the complex spherical harmonics, but I was originally lazy when making the Gradio demo below, and things broke when I updated them to the real [real spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics).
The shapes look pretty similar in any case.)

# 🧠 Equivariant GNNs using Spherical Harmonics

## ⏪ Equivariance Recap

Recall that equivariant graph neural networks (GNNs) are GNNs that behave sensibly when their inputs are transformed according to a group operation e.g. rotations. 

Consider two GNNs both taking as input a point cloud $\mathcal{G}$, where each point is also associated with a colour.
One GNN $f(\mathcal{G})$ labels a point cloud as cat-like or dog-like.
The second GNN $h(\mathcal{G})$ takes the input point cloud and changes the pet to be green.
(Let’s not worry about how to implement these.) Let’s also train these GNNs on upright pets only.

The point is, if I input an upside down cat (implemented through a rotation operator $R$), we would like the first GNN to still label the pet as a cat if it was indeed a cat: $f(R \cdot \mathcal{G}) = f(\mathcal{G})$.
Indeed, this should hold no matter what the group operation $R \in SO(3)$ was.
That is, we want an *invariant* GNN.
For the second GNN, an upside down cat should become an upside down green cat: $h(R \cdot \mathcal{G}) = R \cdot h(\mathcal{G})$.
Such a GNN would be *equivariant*, because the output also transforms under a representation of the same group element.
Invariance is a special case of equivariance with trivial representation.

## 📚 Lore

The first GNN to use spherical harmonics as its building block for constructing equivariant GNNs was [Tensor Field Networks](https://arxiv.org/abs/1802.08219) (TFN), which acted on point clouds (treated as fully connected graphs).
This inspired many other works, arguably the second most famous example being [SE(3) Transformers](https://arxiv.org/abs/2006.10503) which acts on graphs (not just fully connected) and simply adds an attention mechanism during the message passing steps.

## 🔑 Major Key: Tensor Product of Representations

I previously introduced some easy ways to make reducible representations from irreducible ones: taking the Kronecker sum (i.e. putting things on the block diagonal) like $\rho_1(g) \oplus \rho_2(g)$.

We can also take the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) of two representations, like $\rho_1(g) \otimes \rho_2(g)$, which acts on a larger space, the tensor product of the two underlying vector spaces.
The homomorphism relations are again satisfied since the matrix multiplications happen independently (according to the mixed-product property):

$$
[\rho_1(a) \otimes \rho_2(a)][\rho_1(b) \otimes \rho_2(b)] = [\rho_1(a) \rho_1(b)] \otimes [\rho_2(a)\rho_2(b)] =\rho_1(c) \otimes \rho_2(c).
$$

### 💬 Footnote: Kronecker product example

$$
\left[\begin{array}{ll}1 & 2 \\3 & 4\end{array}\right] \otimes\left[\begin{array}{ll}0 & 5 \\6 & 7\end{array}\right]=\left[\begin{array}{cc|cc}1 \times 0 & 1 \times 5 & 2 \times 0 & 2 \times 5 \\1 \times 6 & 1 \times 7 & 2 \times 6 & 2 \times 7 \\\hline 3 \times 0 & 3 \times 5 & 4 \times 0 & 4 \times 5 \\3 \times 6 & 3 \times 7 & 4 \times 6 & 4 \times 7\end{array}\right]=\left[\begin{array}{cc|cc}0 & 5 & 0 & 10 \\6 & 7 & 12 & 14 \\\hline 0 & 15 & 0 & 20 \\18 & 21 & 24 & 28\end{array}\right]
$$

This resulting representation will either be reducible or irreducible.
Without loss of generality we can say that

$$
\rho_1(g) \otimes \rho_2(g)=Q^{-1} \left[ \bigoplus_{i} \rho_{r_i}(g) \right] Q,
$$

where $r_i$ labels a type of irrep on the diagonal.

## 🗝️ Still Kinda Major but Slightly More Boring Key: Clebsch-Gordan Coefficients

We can of course take the $\rho_i(g)$ above to be representations of $SO(3)$, the Wigner $\mathbf{D}^{\ell}(R)$ matrices and work things out from there, but motivated by Tensor Field Networks, let’s instead consider the tensor product of the basis vectors, the spherical harmonics.

Let’s take

$$
\boldsymbol{Y}^{\ell_i}_{m_i}(\mathbf{r}) \otimes \boldsymbol{Y}^{\ell_f}_{m_f}(\mathbf{r}).
$$

I have claimed that this will be the Kronecker sum of some spherical harmonics multiplied by a matrix $Q$, but which spherical harmonics precisely? We know for certain that the result must be an eigenstate of $\boldsymbol{L}_z \otimes \mathbf{1} + \mathbf{1} \otimes \boldsymbol{L}_z$ because of distributivity:

$$
[\boldsymbol{L}_z \otimes \mathbf{1} + \mathbf{1} \otimes \boldsymbol{L}_z] \boldsymbol{Y}^{\ell_i}_{m_i}(\mathbf{r}) \otimes \boldsymbol{Y}^{\ell_f}_{m_f}(\mathbf{r}) = -i(m_i + m_f) \boldsymbol{Y}^{\ell_i}_{m_i}(\mathbf{r}) \otimes \boldsymbol{Y}^{\ell_f}_{m_f}.
$$

Letting $m_o = m_i + m_f$, we see that the maximum this can be is $m_o = \ell_i + \ell_f$, which will happen once (for $m_i = \ell_i$, $m_f = \ell_f$).
The second biggest value it can be is $m_o = \ell_i + \ell_f - 1$, which will happen twice (for $m_i = \ell_i$, $m_f = \ell_f -1$ and for $m_i = \ell_i - 1$, $m_f = \ell_f$).
We can continue this pattern on until $m_o = 0$, which happens $\min(\ell_i,\ell_f)$ times because we always need $m_i = -m_f$.
Recalling that for angular momentum $\ell_o$ we had $m_o \in \{ -\ell_o, -\ell_o + 1, \ldots, \ell_o - 1, \ell_o\}$, we see that we have a maximum of $\ell_o = \ell_i + \ell_f$, and we run out of zeroes by the time we get down to $\ell_o = |\ell_i - \ell_f|$, which is our minimum output angular momentum.

That paragraph was rather unfortunate, but the mnemonic is easy: $|\ell_i - \ell_f| \leq \ell_o \leq \ell_i + \ell_f$, like the triangle inequality of placing two vectors of length $\ell_i$ and $\ell_f$ tip to tail.
Hence we have 

$$
\boldsymbol{Y}^{\ell_i}(\mathbf{r}) \otimes \boldsymbol{Y}^{\ell_f}(\mathbf{r}) = \boldsymbol{C}^{-1} \bigoplus_{\ell_o = |\ell_i - \ell_f|}^{\ell_i + \ell_f} \boldsymbol{Y}^{\ell_o}(\mathbf{r}).
$$

The Clebsch-Gordan coefficients are nothing but the elements of the boring change-of-basis matrix $\boldsymbol{C}$, indexed as $C_{(\ell_i,m_i),(\ell_f,m_f)}^{(\ell_o, m_o)}$.

## 😩 Tensor Field Networks

**Good News**: The basic ideas of TFN are easy to understand once we’re happy with representations, spherical harmonics and tensor products.
Using real spherical harmonics also means that we mostly don’t need to use complex number floating point operations.

**Bad News**: Dealing with latent features that must all be treated differently in the neural network gets finicky, especially when dealing with multiple channels.
This means that notation in TFN has quite a few indices floating about, and keeping track of weights can be slightly annoying.

Essentially, latent node features $\boldsymbol{x}_s^{\ell_i}$ are coefficients multiplying spherical harmonics, and GNN messages are passed by taking their tensor product with spherical harmonic embeddings of displacement vectors $\boldsymbol{Y}^{\ell_f}(\mathbf{r}_{st})$ (multiplied by some learnable components).

## 🍺 WIP Implementation of TFN

If you’re interested in an unfinished, unpolished, undocumented barebones implementation of TFN in JAX from someone who’s never used JAX before, then boy do I have the repository for you…

https://github.com/ElisR/EquiformerFlux

Specifically, the [Tetris example](https://github.com/ElisR/EquiformerFlux/blob/main/equiformer/examples/tetris.py) shows how to construct an equivariant Tetris shape classifier that only gets trained on one orientation of each shape.
Also of interest may be the `TFNLayer` module (in `layers.py`), and functions for calculating spherical harmonics and tensor products in `spherical.py` and `tensor_product.py`, respectively.

One cute part of this repository is that reasonably-efficient JAX implementations of spherical harmonics are computed on the fly (without being hardcoded in like in `e3nn`) through metaprogramming.
This happens by using the computer algebra of SymPy to generate simplified real spherical harmonics in terms of Cartesian coordinates, which can then be compiled into JAX functions.
(To me this is quite a bit simpler than the $SE(3)$ Transformers’ appendix about executing recurrence relations with dynamic programming on the GPU.)

What’s not so cute is how spherical harmonics are recomputed many times by individual neural network layers, even though they could be reused (and a similar story holds for Clebsch-Gordan coefficients).
At least this recalculation makes it easier to read for pedagogical purposes, but I may update this in the future to make it more efficient.

## 🔮 The Future of TFN

In some ways TFN is beautiful. In other ways, it is quite ugly.

When implementing TFN, one of the ugliest things is the fact that each feature with different angular momentum has a different number of components, which means one has to be careful with how they mix together.
(Indeed, in my implementation I have kept different feature vectors as separate elements in a dictionary to avoid the headache.)
Concatenating everything in one big tensor that can be efficiently operated on requires having a very intricate indexing scheme (which I gave up on for this talk).

Another non-beautiful thing is that when converting displacement vectors to spherical harmonics, one always has $z$ as a “special” axis, even though it is an arbitrary direction.
This is fine because everything in the network is self-consistently equivariant thereafter, but it doesn’t feel spiritually equivariant.

As for something that is more than just aesthetics, swapping from cartesian components to spherical harmonics and performing tensor products for the large-$\ell$ representations adds up to a lot of floating point operations.
Having to store all the $2\ell +1$-sized feature vectors also puts a limit on how high one can go with $\ell$.
Going to large $\ell$ is important for learning fine angular information (see [GNN expressivity paper](https://arxiv.org/abs/2301.09308)), so this is especially unfortunate.

This is why [Passaro and Zitnick’s improvement](https://arxiv.org/abs/2302.03655) is very cool!
They simplify every aspect of TFN by noticing it is better to have the arbitrary “special” axis not be arbitrary, and rather have it match the axis along which messages are being sent (i.e. the displacement vector between neighbouring nodes).
This makes everything much more sparse and efficient.
This has already been implemented in some modern architectures like [EquiformerV2](https://arxiv.org/abs/2306.12059), and will probably soon replace TFN and $SE(3)$ Transformers everywhere they have been used thus far.
