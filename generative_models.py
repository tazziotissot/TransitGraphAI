import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GCNConv(nn.Module):
    """
    Graph Convolutional layer with pre-normalized adjacency.

    Parameters
    ----------
    in_features : int
        Input feature dimension F_in.
    out_features : int
        Output feature dimension F_out.

    Notes
    -----
    This layer expects the adjacency matrix `adj` to already include self-loops.
    It performs the standard GCN propagation:  D^{-1/2} A D^{-1/2} X W.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [N, F_in].
        adj : torch.Tensor
            Symmetric adjacency matrix with self-loops of shape [N, N].

        Returns
        -------
        torch.Tensor
            Output node embeddings of shape [N, F_out].
        """

        # Compute D^(-1/2)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        # Compute normalized adjacency
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt  # [N, N]

        # GCN propagation
        out = adj_norm @ x  # [N, F]
        out = self.linear(out)  # [N, out_features]
        return out


class PureGCNConv(nn.Module):
    """
    Graph Convolutional layer that adds self-loops and applies symmetric normalization.

    Parameters
    ----------
    in_features : int
        Input feature dimension F_in.
    out_features : int
        Output feature dimension F_out.

    Notes
    -----
    This layer internally adds I to the adjacency and computes
    D^{-1/2} (A + I) D^{-1/2} X W.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [N, F_in].
        adj : torch.Tensor
            Adjacency matrix (self-loops may be missing) of shape [N, N].

        Returns
        -------
        torch.Tensor
            Output node embeddings of shape [N, F_out].
        """
        # Adding self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)

        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        # Propagation
        out = adj_norm @ x
        out = self.linear(out)
        return out


class GCNEncoder(nn.Module):
    """
    GCN-based encoder that outputs mean and log-variance for a latent Gaussian.

    Parameters
    ----------
    in_channels : int
        Input feature dimension F_in.
    hidden_channels : int
        Hidden dimension H.
    latent_dim : int
        Latent dimension Z.

    Notes
    -----
    The encoder uses a shared first GCN layer followed by two heads (mu/logvar),
    each implemented with a GCN layer (PureGCNConv).
    """

    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = PureGCNConv(in_channels, hidden_channels)
        self.conv_mu = PureGCNConv(hidden_channels, latent_dim)
        self.conv_logvar = PureGCNConv(hidden_channels, latent_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [N, F_in].
        edge_index : torch.Tensor
            Adjacency matrix of shape [N, N].

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Tuple (mu, logvar) with shapes [N, Z] each.
        """
        h = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar


class InnerProductDecoder(nn.Module):
    """
    Inner-product decoder that reconstructs an adjacency matrix from latent node embeddings.

    Notes
    -----
    Produces a dense, symmetric reconstruction via sigma(ZZ^T).
    Useful for link prediction and graph auto-encoding/variational auto-encoding.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        z : torch.Tensor
            Latent node embeddings of shape [N, Z].

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency probabilities in [0, 1] with shape [N, N].
        """
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_pred


class GraphVAE(nn.Module):
    """
    Variational Graph Autoencoder with a GCN encoder and inner-product decoder.

    Parameters
    ----------
    in_channels : int
        Input feature dimension F_in.
    hidden_channels : int
        Hidden dimension H in the encoder.
    latent_dim : int
        Latent dimension Z.

    Notes
    -----
    The model encodes nodes into (mu, logvar), samples z via the reparameterization
    trick, and reconstructs adjacency with an inner-product decoder.
    """

    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, latent_dim)
        self.decoder = InnerProductDecoder()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample latent embeddings using the reparameterization trick.

        Parameters
        ----------
        mu : torch.Tensor
            Mean tensor of shape [N, Z].
        logvar : torch.Tensor
            Log-variance tensor of shape [N, Z].

        Returns
        -------
        torch.Tensor
            Sampled latent embeddings z of shape [N, Z].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [N, F_in].
        adj : torch.Tensor
            Adjacency matrix of shape [N, N].

        Returns
        -------
        (torch.Tensor, torch.Tensor, torch.Tensor)
            Tuple (adj_recon, mu, logvar):
            - adj_recon: reconstructed adjacency probabilities [N, N]
            - mu, logvar: encoder outputs [N, Z]
        """
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        adj_recon = self.decoder(z)
        return adj_recon, mu, logvar


def loss_function(
    adj_recon: torch.Tensor,
    adj_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    VAE loss for graph reconstruction: binary cross-entropy + KL divergence.

    Parameters
    ----------
    adj_recon : torch.Tensor
        Reconstructed adjacency probabilities in [0, 1], shape [N, N].
    adj_true : torch.Tensor
        Binary ground-truth adjacency matrix, shape [N, N].
    mu : torch.Tensor
        Mean from the encoder, shape [N, Z].
    logvar : torch.Tensor
        Log-variance from the encoder, shape [N, Z].

    Returns
    -------
    torch.Tensor
        Scalar loss tensor (BCE reconstruction loss + KL divergence).
    """
    ...
    recon_loss = F.binary_cross_entropy(adj_recon, adj_true, reduction="mean")
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def generate_batched_graphs(
    num_graphs: int,
    N: int,
    p: float = 0.1,
    use_eye: bool = True,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of undirected Erdős–Rényi graphs and simple node features.

    Parameters
    ----------
    num_graphs : int
        Number of graphs B to generate.
    N : int
        Number of nodes per graph.
    p : float, default=0.1
        Edge probability for G(N, p).
    use_eye : bool, default=True
        If True, node features are an identity matrix (one-hot) of shape [N, N].
        If False, features are constant ones of shape [N, 1].
    seed : int | None, optional
        Numpy random seed for reproducibility.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        Tuple (adjs, features):
        - adjs: adjacency tensors of shape [B, N, N] with 0/1 entries (upper-tri symmetrized)
        - features: feature tensors of shape [B, N, F] where F=N if `use_eye` else 1

    Notes
    -----
    - Graphs are generated without self-loops.
    - Adjacency is made symmetric from the upper triangular part.
    """
    if seed is not None:
        np.random.seed(seed)

    adjs = []
    features = []

    for _ in range(num_graphs):
        # Generate a symmetric adjacency matrix
        upper = np.triu(np.random.rand(N, N) < p, 1).astype(np.float32)
        adj = upper + upper.T
        adjs.append(torch.tensor(adj))

        # Generate features
        if use_eye:
            x = torch.eye(N)  # One-hot (single for each vertex)
        else:
            x = torch.ones((N, 1))  # Identical for all vertices

        features.append(x)

    # Stacking them in batches
    adjs = torch.stack(adjs)  # [B, N, N]
    features = torch.stack(features)  # [B, N, F]

    return adjs, features


class GraphDiscriminator(nn.Module):
    """
    Discriminator network for adversarial training on graphs.

    Parameters
    ----------
    N : int
        Number of nodes per graph (the adjacency is flattened to length N*N).

    Notes
    -----
    The discriminator flattens each adjacency matrix [B, N, N] into a vector
    [B, N*N] and passes it through two fully connected layers with LeakyReLU
    activations, ending in a sigmoid output for real/fake probability.
    """

    def __init__(self, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N * N, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        adj : torch.Tensor
            Batch of adjacency matrices of shape [B, N, N].

        Returns
        -------
        torch.Tensor
            Real/fake probabilities of shape [B, 1].
        """
        return self.net(adj)


class GraphGenerator(nn.Module):
    """
    Generator network producing adjacency matrices from latent vectors.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent input vector.
    N : int
        Number of nodes per generated graph.

    Notes
    -----
    - Outputs symmetric adjacency matrices with values in [0, 1].
    - The diagonal is set to zero (no self-loops).
    - Acts as a simple MLP generator mapping z -> σ(W₂ ReLU(W₁ z)).
    """

    def __init__(self, latent_dim, N):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, N * N),
            nn.Sigmoid(),  # In order to keep values between 0 and 1 (edge likelihoods)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors of shape [B, latent_dim].

        Returns
        -------
        torch.Tensor
            Generated symmetric adjacency matrices of shape [B, N, N],
            with entries in [0, 1].
        """
        out = self.net(z)  # [B, N*N]
        adj = out.view(-1, self.N, self.N)  # [B, N, N]
        # Making a symmetrical matrix
        adj_sym = 0.5 * (adj + adj.transpose(1, 2))
        # Optional: we can force the diagonal to zero
        adj_sym = adj_sym * (1 - torch.eye(self.N, device=adj.device).unsqueeze(0))
        return adj_sym


def trainGraphGAN(
    N: int,
    hidden: int,
    latent: int,
    num_epochs: int,
    B: int,
    num_batches: int,
    cycle_d: int,
    is_vae: bool,
):
    """
    Train either a standalone GraphGAN or a hybrid VAE–GAN on synthetic graphs.

    Parameters
    ----------
    N : int
        Number of nodes per graph.
    hidden : int
        Hidden dimension for the encoder or generator network.
    latent : int
        Latent dimension Z for the generator or VAE.
    num_epochs : int
        Number of training epochs.
    B : int
        Batch size (number of graphs per iteration).
    num_batches : int
        Number of batches per epoch.
    cycle_d : int
        Update frequency for the discriminator; it is trained every `cycle_d` epochs.
    is_vae : bool
        If True, uses a GraphVAE as generator (hybrid VAE–GAN).
        If False, uses a simple GraphGenerator.

    Returns
    -------
    (nn.Module, nn.Module)
        Tuple (generator_or_vae, discriminator) trained on synthetic graphs.

    Notes
    -----
    - Real graphs are sampled from `generate_batched_graphs` and corrupted with Gaussian noise.
    - The discriminator is trained on real vs. generated graphs using BCE loss with label smoothing.
    - The generator is optimized to fool the discriminator, optionally combined with a VAE reconstruction + KL term.
    - Optimizers: AdamW with learning rate 1e-3 and weight decay 1e-2.
    - Prints per-epoch discriminator and generator losses.
    """
    if is_vae:
        model = GraphVAE(in_channels=N, hidden_channels=hidden, latent_dim=latent)
    else:
        model = GraphGenerator(latent, N)
    disc = GraphDiscriminator(N)
    optimizer_G = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    optimizer_D = optim.AdamW(disc.parameters(), lr=1e-3, weight_decay=1e-2)

    for e in range(num_epochs):
        tot_loss_d = 0
        tot_loss_g = 0
        update_G = True
        update_D = (
            e % cycle_d == 0
        )  # Desynchronisation of trainings: the generator is trained five times more often than the discriminator
        for b in range(num_batches):
            model.train()
            disc.train()
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # Generation of reference graphs
            adj_real, x = generate_batched_graphs(B, N)
            adj_real = (
                adj_real + torch.randn_like(adj_real) * 0.1
            )  # Adding noise to reference adjacency matrix

            # Forward VAE
            if is_vae:
                adj_fake, mu, logvar = model(x, adj_real)
            else:
                z = torch.randn(B, latent)
                adj_fake = model(z)
            # --- Discriminator ---
            pred_real = disc(adj_real)
            pred_fake = disc(adj_fake.detach())

            # Labels
            real_labels = (
                torch.rand_like(pred_real) * 0.3 + 0.7
            )  # Label smoothing in order to increase the discriminator uncertainty
            fake_labels = torch.rand_like(pred_fake) * 0.3

            loss_disc = F.binary_cross_entropy(
                pred_real, real_labels
            ) + F.binary_cross_entropy(pred_fake, fake_labels)
            tot_loss_d += loss_disc.item()
            if update_D:
                loss_disc.backward()
                optimizer_D.step()

            # --- Generator (VAE + GAN loss) ---
            pred_fake_for_gen = disc(adj_fake)
            loss_gan_gen = F.binary_cross_entropy(pred_fake_for_gen, real_labels)

            if is_vae:
                loss_vae = loss_function(adj_fake, adj_real, mu, logvar)
                loss_total = loss_vae + 0.1 * loss_gan_gen
            else:
                loss_total = loss_gan_gen
            tot_loss_g += loss_total.item()
            if update_G:
                loss_total.backward()
                optimizer_G.step()
        print(
            f"Epoch {e+1}: loss discriminator {tot_loss_d:.2f}, loss generator {tot_loss_g:.2f}"
        )
    return model, disc
