import torch
import random

def gen_share(X, type="float"):
    if type == "int":
        tmp_x = random.randint(-2<<16, 2<<16)
        X_0 = tmp_x
        X_1 = X - tmp_x
    elif type == "bool":
        rnd = round(random.random())
        if X == 0:
            if rnd == 0:
                X_0 = 0
                X_1 = 0
            else:
                X_0 = 1
                X_1 = 1
        else:
            if rnd == 0:
                X_0 = 1
                X_1 = 0
            else:
                X_0 = 0
                X_1 = 1
    else:
        tmp_x = torch.randn(X.shape)
        X_0 = tmp_x
        X_1 = X - X_0
    # print('X_0 = ', X_0)
    # print('X_1 = ', X_1)
    return X_0, X_1

def matmul(X, Y):
    X_0, X_1 = X[0], X[1]
    Y_0, Y_1 = Y[0], Y[1]
    A = torch.randn(X_0.shape)
    B = torch.randn(Y_0.shape)
    C = torch.matmul(A, B)
    A_0, A_1 = gen_share(A)
    B_0, B_1 = gen_share(B)
    C_0, C_1 = gen_share(C)

    E_0 = X_0 - A_0
    F_0 = Y_0 - B_0
    E_1 = X_1 - A_1
    F_1 = Y_1 - B_1

    E = E_0 + E_1
    F = F_0 + F_1

    P_0 = torch.matmul(X_0, F) + torch.matmul(E, Y_0) + C_0
    P_1 = -torch.matmul(E, F) + torch.matmul(X_1, F) + torch.matmul(E, Y_1) + C_1

    return P_0, P_1

def test_matmul(X, Y):
    no_mpc = torch.matmul(X, Y)
    X_0, X_1 = gen_share(X)
    Y_0, Y_1 = gen_share(Y)
    S_0, S_1 = matmul([X_0, X_1], [Y_0, Y_1])
    mpc = S_0 + S_1
    print('torch.matmul(X, Y) = ', no_mpc)
    print('MPC MatMul X dot Y = ', mpc)
    # print('validate result(no_mpc == mpc): ', no_mpc == mpc)

def select_share(X, Y, alpha):
    X_0, X_1 = X[0], X[1]
    Y_0, Y_1 = Y[0], Y[1]
    A_0, A_1 = gen_share(alpha) 

    W_0 = Y_0 - X_0
    W_1 = Y_1 - X_1

    C_0, C_1 = matmul([A_0, A_1], [W_0, W_1])

    Z_0 = X_0 + C_0
    Z_1 = X_1 + C_1

    return Z_0, Z_1


def test_select_share(X, Y, alpha):
    no_mpc = (1 - alpha) * X + alpha * Y
    X_0, X_1 = gen_share(X)
    Y_0, Y_1 = gen_share(Y)
    mpc_0, mpc_1 = select_share([X_0, X_1], [Y_0, Y_1], alpha)
    print('select_share(X, Y) = ', no_mpc)
    print('MPC select_share (X, Y) = ', mpc_0 + mpc_1)


def private_compare(X, R, beta=0, L=32):
    X_0, X_1 = X[0], X[1]
    T = (R + 1) % (1 << L - 1)
    beta = beta % 2

    # print('X = ', X_0 + X_1)
    # print('R = ', R)
    # print('beta = ', beta)
    # print('X > R = ', (X_0 + X_1 > R))
    # print('beta xor (X > R) = ', beta ^ (X_0 + X_1 > R))

    t_X_0 = X_0
    t_X_1 = X_1
    t_R = R
    t_T = T

    bitx_0 = []
    bitx_1 = []
    bitr = []
    bitt = []
    for _ in range(L):
        bitx_0.append(t_X_0 % 2)
        bitx_1.append(t_X_1 % 2)
        t_X_0 = int(t_X_0 / 2)
        t_X_1 = int(t_X_1 / 2)

        bitr.append(t_R % 2)
        t_R = int(t_R / 2)
        bitt.append(t_T % 2)
        t_T = int(t_T / 2)
    bitx_0.reverse()
    bitx_1.reverse()
    bitr.reverse()
    bitt.reverse()

    # print('X_0 = ', X_0, ', bin(X_0) = ', bin(X_0))
    # print(bitx_0)
    # print('X_1 = ', X_1, ', bin(X_1) = ', bin(X_1))
    # print(bitx_1)
    # print('bin(R) = ', bin(R))
    # print(bitr)
    # print('bin(T) = ', bin(T))
    # print(bitt)

    w_0 = []
    w_1 = []
    c_0 = []
    c_1 = []
    for i in range(L - 1, -1, -1):
        if beta == 0:
            w_i_0 = bitx_0[i] - 2 * bitr[i] * bitx_0[i]
            w_i_1 = bitx_1[i] - 2 * bitr[i] * bitx_1[i] + bitr[i]
            w_0.append(w_i_0)
            w_1.append(w_i_1)

            c_i_0 = -bitx_0[i] + sum(w_0) - w_i_0
            c_i_1 = -bitx_1[i] + sum(w_1) - w_i_1 + 1 + bitr[i]
            c_0.append(c_i_0)
            c_1.append(c_i_1)
        elif beta == 1 and R != (2 << L - 1):
            w_i_0 = bitx_0[i] - 2 * bitt[i] * bitx_0[i]
            w_i_1 = bitx_1[i] - 2 * bitt[i] * bitx_1[i] + bitt[i]
            w_0.append(w_i_0)
            w_1.append(w_i_1)

            c_i_0 = bitx_0[i] + sum(w_0) - w_i_0
            c_i_1 = bitx_1[i] + sum(w_1) - w_i_1 + 1 - bitt[i]
            c_0.append(c_i_0)
            c_1.append(c_i_1)
        else:
            if i != 1:
                c_i_0 = 1
                c_i_1 = 0
            else:
                c_i_0 = 1
                c_i_1 = -1
            c_0.append(c_i_0)
            c_1.append(c_i_1)

    idx = [i for i in range(L)]
    random.shuffle(idx)
    # print(idx)

    permute_c_0 = []
    permute_c_1 = []
    for i in range(len(c_0)):
        permute_c_0.append(c_0[idx[i]])
        permute_c_1.append(c_1[idx[i]])
    # # print('c_0=')
    # # print(c_0)
    # # print('c_1=')
    # # print(c_1)
    # # print('permute_c_0=')
    # # print(permute_c_0)
    # # print('permute_c_1=')
    # # print(permute_c_1)

    # tmp = [w_0[i] + w_1[i] for i in range(len(w_0))]
    # print('W =\n', tmp)

    c = []
    for i in range(len(permute_c_0)):
        c.append(permute_c_0[i] + permute_c_1[i])
    # print('c=')
    # print(c)
    # for i in range(len(c_0)):
    #     c.append(c_0[i] + c_1[i])
    # print('c=')
    # print(c)
    signal = False
    for it in c:
        if it == 0:
            signal = True
    if signal:
        return 1
    else:
        return 0

def test_private_compare(X, R, beta=0, L=32):
    X_0, X_1 = X[0], X[1]
    label = beta ^ ((X_0 + X_1) > R)
    res = private_compare([X_0, X_1], R, beta)
    # print('beta xor (X > R) = ', label)
    # print('private_compare = ', res)
    if res != label:
        print('beta = ', beta, ', X = ', X, ', R = ', R)

def wrap(x, y, L):
    return (1 if x + y >= L else 0)

def share_convert(A, L=1<<31):
    U_0, U_1 = gen_share(0, "int")
    A_0, A_1 = A[0], A[1]
    R = random.randint(0, 1<<16)
    R_0, R_1 = gen_share(R, "int")

    alpha = wrap(R_0, R_1, L)

    H_A_0 = A_0 + R_0
    H_A_1 = A_1 + R_1

    beta_0 = wrap(A_0, R_0, L)
    beta_1 = wrap(A_1, R_1, L)

    X = H_A_0 + H_A_1
    delta = wrap(H_A_0, H_A_1, L)

    X_0, X_1 = gen_share(X, "int")


    delta_0, delta_1 = gen_share(delta, "bool")

    yeta2 = round(random.random())

    yeta1 = private_compare([X_0, X_1], R - 1, yeta2)
    yeta1_0, yeta1_1 = gen_share(yeta1, "bool")

    yeta_0 = yeta1_0 - 2 * yeta2 * yeta1_0 + yeta2
    yeta_1 = yeta1_1 - 2 * yeta2 * yeta1_1

    theta_0 = beta_0 + delta_0 + yeta_0 - alpha - 1
    theta_1 = beta_1 + delta_1 + yeta_1

    y_0 = A_0 - theta_0 + U_0
    y_1 = A_1 - theta_1 + U_1

    return y_0, y_1


def test_share_convert(A, L=1<<31):
    print('test_share_convert')
    # res_0, res_1 = share_convert(A, L)
    # print('A = ', A[0] + A[1])
    # print('share_convert(A=', A, ', L=', L,') = ', res_0 + res_1)
    print('A = ', A[0] + A[1])
    tmp = []
    for i in range(20):
        res_0, res_1 = share_convert(A, L)
        print('share_convert(A=', A, ', L=', L,') = ', res_0 + res_1)
        tmp.append(res_0 + res_1)
    print(tmp)



if 1:
    # beta = 0
    # X_0 = 1311
    # X_1 = -1000
    R = 310
    for beta in ([0, 1]):
        for X_0 in range(301, 321):
            test_private_compare([X_0, 0], R, beta, L=32)


    # test_private_compare([X_0, X_1], R, beta=0, L=32)

if 0:
    A_0 = 14
    A_1 = 14
    test_share_convert([A_0, A_1], L=1<<31)