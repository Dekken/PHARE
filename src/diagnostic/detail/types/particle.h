#ifndef PHARE_DIAGNOSTIC_DETAIL_TYPES_PARTICLE_H
#define PHARE_DIAGNOSTIC_DETAIL_TYPES_PARTICLE_H

template<typename HighFiveDiagnostic>
class ParticlesDiagnosticWriter : public Hi5DiagnosticWriter<HighFiveDiagnostic>
{
public:
    using Hi5DiagnosticWriter<HighFiveDiagnostic>::hi5_;
    using Attributes                = typename Hi5DiagnosticWriter<HighFiveDiagnostic>::Attributes;
    static constexpr auto dimension = HighFiveDiagnostic::dimension;
    ParticlesDiagnosticWriter(HighFiveDiagnostic& hi5)
        : Hi5DiagnosticWriter<HighFiveDiagnostic>(hi5)
    {
    }
    void getDataSetInfo(std::string const& patchID, Attributes& patchAttributes) override;
    void initDataSets(std::vector<std::string> const& patchIDs,
                      Attributes& patchAttributes) override;
    void write(DiagnosticDAO&) override;
    void compute(DiagnosticDAO&) override{};

private:
    size_t maxPops = 0, level_ = 0;
    std::unique_ptr<ParticlePacker<dimension>> packer;
};

template<typename HighFiveDiagnostic>
void ParticlesDiagnosticWriter<HighFiveDiagnostic>::getDataSetInfo(std::string const& patchID,
                                                                   Attributes& patchAttributes)
{
    auto& outer = this->hi5_;

    auto getSize = [&outer](auto const& value) {
        using ValueType = std::decay_t<decltype(value)>;
        if constexpr (is_array_dataset<ValueType, outer.dimension>)
            return value.size();
        else
            return 1; /* not an array so value one of type ValueType*/
    };

    auto particleInfo = [&](auto& attributes, auto& particles) {
        size_t part_idx = 0;
        packer          = std::make_unique<PHARE::ParticlePacker<dimension>>(particles);
        std::apply(
            [&](auto&... args) {
                (
                    [&]() {
                        attributes[packer->keys()[part_idx]] = getSize(args) * particles.size();
                        part_idx++;
                    }(),
                    ...);
            },
            packer->empty());
    };

    size_t pops = 0;
    for (auto& pop : outer.modelView().getIons())
    {
        std::string popIdx                       = "pop_" + std::to_string(pops);
        patchAttributes[patchID][popIdx]["name"] = pop.name();
        particleInfo(patchAttributes[patchID][popIdx]["domain"], pop.domainParticles());
        particleInfo(patchAttributes[patchID][popIdx]["levelGhost"], pop.levelGhostParticles());
        particleInfo(patchAttributes[patchID][popIdx]["patchGhost"], pop.patchGhostParticles());
        pops++;
    }
    maxPops = pops;
}

template<typename HighFiveDiagnostic>
void ParticlesDiagnosticWriter<HighFiveDiagnostic>::initDataSets(
    std::vector<std::string> const& patchIDs, Attributes& patchAttributes)
{
    auto& outer = this->hi5_;

    auto createDataSet = [&outer](auto&& path, auto size, auto const& value) {
        using ValueType = std::decay_t<decltype(value)>;
        if constexpr (is_array_dataset<ValueType, outer.dimension>)
            return outer.template createDataSet<typename ValueType::value_type>(path, size);
        else
            return outer.template createDataSet<ValueType>(path, size);
    };

    auto initDataSet = [&](auto& patchID, auto name, auto& attributes, bool empty) {
        std::string path(hi5_.getPatchPath("time", level_, patchID) + "/ions/pop/" + name + "/");
        size_t part_idx = 0;
        std::apply(
            [&](auto&... args) {
                (
                    [&]() {
                        createDataSet(
                            path + packer->keys()[part_idx],
                            empty ? 0 : attributes[packer->keys()[part_idx]].template to<size_t>(),
                            args);
                        part_idx++;
                    }(),
                    ...);
            },
            packer->empty());
    };

    auto initPop = [&](auto& attributes, std::string patchID = "", std::string idx = "",
                       std::string name = "") {
        bool empty = patchID.empty();
        initDataSet(patchID, name + "/domain", empty ? attributes : attributes[idx]["domain"],
                    empty);
        initDataSet(patchID, name + "/levelGhost",
                    empty ? attributes : attributes[idx]["levelGhost"], empty);
        initDataSet(patchID, name + "/patchGhost",
                    empty ? attributes : attributes[idx]["patchGhost"], empty);
    };

    auto initNullPatch = [&]() {
        for (size_t i = 0; i < maxPops; i++)
        {
            initPop(patchAttributes);
        }
    };

    auto initPatch = [&](auto& patchID, auto& attributes) {
        for (size_t i = 0; i < maxPops; i++)
        {
            std::string popIdx  = "pop_" + std::to_string(i);
            std::string popName = attributes[popIdx]["name"].template to<std::string>();
            initPop(attributes, patchID, popIdx, popName);
        }
    };

    size_t patches    = hi5_.patchIDs().size();
    size_t maxPatches = hi5_.getMaxOfPerMPI(patches);
    for (size_t i = 0; i < patches; i++)
    {
        initPatch(patchIDs[i], patchAttributes[patchIDs[i]]);
    }
    for (size_t i = patches; i < maxPatches; i++)
    {
        initNullPatch();
    }
}

template<typename HighFiveDiagnostic>
void ParticlesDiagnosticWriter<HighFiveDiagnostic>::write([
    [maybe_unused]] DiagnosticDAO& diagnostic)
{
    auto& outer = this->hi5_;

    auto writeDataSet = [&outer](auto path, auto& start, auto const& value) {
        using ValueType = std::decay_t<decltype(value)>;
        if constexpr (is_array_dataset<ValueType, outer.dimension>)
            outer.writeDataSetPart(path, start * value.size(), value.size(), value);
        else
            outer.writeDataSetPart(path, start, 1, value); /*not array, write 1 value*/
    };

    auto writeParticles = [&](auto path, auto& particles) {
        if (!particles.size())
            return;

        packer = std::make_unique<PHARE::ParticlePacker<dimension>>(particles);

        size_t part_idx = 0, idx = 0;
        while (packer->hasNext())
        {
            part_idx = 0;
            std::apply(
                [&](auto&... args) {
                    (
                        [&]() {
                            writeDataSet(path + packer->keys()[part_idx], idx, args);
                            part_idx++;
                        }(),
                        ...);
                },
                packer->next());
            idx++;
        }
    };

    for (auto& pop : outer.modelView().getIons())
    {
        std::string path(outer.patchPath() + "/ions/pop/" + pop.name() + "/");
        writeParticles(path + "domain/", pop.domainParticles());
        writeParticles(path + "levelGhost/", pop.levelGhostParticles());
        writeParticles(path + "patchGhost/", pop.patchGhostParticles());
    }
}

#endif /* PHARE_DIAGNOSTIC_DETAIL_TYPES_PARTICLE_H */