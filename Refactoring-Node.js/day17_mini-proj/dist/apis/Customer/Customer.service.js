"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
var __rest = (this && this.__rest) || function (s, e) {
    var t = {};
    for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0)
        t[p] = s[p];
    if (s != null && typeof Object.getOwnPropertySymbols === "function")
        for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
            if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i]))
                t[p[i]] = s[p[i]];
        }
    return t;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CustomerService = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const typeorm_2 = require("typeorm");
const Customer_entity_1 = require("./entities/Customer.entity");
let CustomerService = class CustomerService {
    constructor(customerRepository) {
        this.customerRepository = customerRepository;
    }
    async create({ createCustomerInput }) {
        const { email } = createCustomerInput, rest = __rest(createCustomerInput, ["email"]);
        const isValid = await this.customerRepository.findOne({
            where: { email: email },
        });
        return isValid
            ? {
                message: await this.customerRepository.save(Object.assign({}, createCustomerInput)),
            }
            : { message: '이미 가입된 이메일입니다.' };
    }
    async findOne({ customerName }) {
        const result = await this.customerRepository.findOne({
            where: { customer_name: customerName },
        });
        return result;
    }
    async findAll() {
        return await this.customerRepository.find();
    }
    async update({ customerId, updateCustomerInput }) {
        const mycustomer = await this.customerRepository.findOne({
            where: { customer_id: customerId },
        });
        const newCustomer = Object.assign(Object.assign(Object.assign({}, mycustomer), { id: customerId }), updateCustomerInput);
        return await this.customerRepository.save(newCustomer);
    }
    async delete({ customerId }) {
        const result = await this.customerRepository.softDelete({
            customer_id: customerId,
        });
        return result.affected
            ? { message: `${customerId}}님, 성공적으로 탈퇴하셨습니다.` }
            : false;
    }
};
CustomerService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, typeorm_1.InjectRepository)(Customer_entity_1.Customer)),
    __metadata("design:paramtypes", [typeorm_2.Repository])
], CustomerService);
exports.CustomerService = CustomerService;
//# sourceMappingURL=Customer.service.js.map