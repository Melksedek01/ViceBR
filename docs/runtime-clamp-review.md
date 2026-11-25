# Runtime: clamp indireto não destrutivo

## Motivo

O caminho antigo de `ExecuteIndirect` atualizava uma sombra de CPU e aplicava
o clamp no próprio recurso de argumentos. Quando um compute anterior na mesma
command list já tinha produzido esses argumentos, essa sequência podia apagar
o resultado recente da GPU antes do consumidor.

## Mudança

- Cada `HostList` mantém um buffer UAV de scratch reutilizável.
- O copy-clamp lê argumentos e contador pelos bindings UAV já compatíveis com
  os produtores do título, sem fazer stores nessas fontes.
- O shader grava registros limitados no scratch; somente o scratch é passado
  para `ExecuteIndirect` no offset zero.
- O scratch usa a sequência de estados `COMMON -> UNORDERED_ACCESS ->
  INDIRECT_ARGUMENT -> COMMON`.
- O código valida alinhamento, stride mínimo por tipo e tamanho antes de
  gravar comandos.

Isso preserva os argumentos originais e elimina a mutação local pelo runtime.
Não é uma prova de que todos os bloqueios seguintes de renderização estejam
resolvidos; as capturas pré/pós por consumidor e os testes de remoção de
dispositivo continuam sendo o próximo diagnóstico.

## Verificação

```text
cargo test -p victory-gpu --lib --quiet
145 passed; 0 failed; 2 ignored

cargo build --release -p victory --features diagnostics
Finished `release` profile [optimized + debuginfo]
```

## Limites de segurança

Esta alteração não contém dados, executáveis, chaves ou quaisquer arquivos do
título. A execução continua sendo responsabilidade do ambiente local e deve
usar o isolamento de rede configurado pelo operador.

## Triagem de remoção de dispositivo

Para diagnóstico somente, `VICTORY_TRIAGE_SKIP_INDIRECT_ADDRESS=0x...` suprime
um consumidor indirect único cujo endereço de argumentos coincide exatamente.
O log o marca como `TRIAGE`; não é uma correção e não deve ser usado como
configuração normal. Ele existe para confirmar uma ocorrência apontada por
DRED sem suprimir as demais ocorrências do mesmo pipeline.
